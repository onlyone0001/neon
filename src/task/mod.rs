//! Asynchronous background _tasks_ that run in the Node thread pool.

use std::marker::{Send, Sized};
use std::mem;
use std::os::raw::c_void;
use std::marker::PhantomData;

use types::{Value, JsFunction, JsUndefined};
use result::JsResult;
use handle::{Handle, Managed};
use context::{Context, TaskContext};
use neon_runtime;
use neon_runtime::raw;

trait BaseTask: Send + Sized + 'static {
    type Output: Send + 'static;
    type JsEvent: Value;

    fn perform(self) -> Self::Output;
    fn complete<'a>(cx: TaskContext<'a>, result: Self::Output) -> JsResult<Self::JsEvent>;

    fn schedule(self, callback: Handle<JsFunction>) {
        let boxed_self = Box::new(self);
        let self_raw = Box::into_raw(boxed_self);
        let callback_raw = callback.to_raw();
        unsafe {
            neon_runtime::task::schedule(mem::transmute(self_raw),
                                         perform_task::<Self>,
                                         complete_task::<Self>,
                                         callback_raw);
        }
    }
}

/// A Rust task that can be executed in a background thread.
pub trait Task: Send + Sized + 'static {
    /// The task's result type, which is sent back to the main thread to communicate a successful result back to JavaScript.
    type Output: Send + 'static;

    /// The task's error type, which is sent back to the main thread to communicate a task failure back to JavaScript.
    type Error: Send + 'static;

    /// The type of JavaScript value that gets produced to the asynchronous callback on the main thread after the task is completed.
    type JsEvent: Value;

    /// Perform the task, producing either a successful `Output` or an unsuccessful `Error`. This method is executed in a background thread as part of libuv's built-in thread pool.
    fn perform(&self) -> Result<Self::Output, Self::Error>;

    /// Convert the result of the task to a JavaScript value to be passed to the asynchronous callback. This method is executed on the main thread at some point after the background task is completed.
    fn complete<'a>(self, cx: TaskContext<'a>, result: Result<Self::Output, Self::Error>) -> JsResult<Self::JsEvent>;

    /// Schedule a task to be executed on a background thread.
    ///
    /// `callback` should have the following signature:
    ///
    /// ```js
    /// function callback(err, value) {}
    /// ```
    fn schedule(self, callback: Handle<JsFunction>) {
        BaseTask::schedule(self, callback)
    }
}

impl<T: Task> BaseTask for T {
    type Output = (Result<T::Output, T::Error>, T);
    type JsEvent = T::JsEvent;

    fn perform(self) -> Self::Output {
        (Task::perform(&self), self)
    }

    fn complete<'a>(cx: TaskContext<'a>, output: Self::Output) -> JsResult<Self::JsEvent> {
        let (result, task) = output;

        task.complete(cx, result)
    }
}

unsafe extern "C" fn perform_task<T: BaseTask>(task: *mut c_void) -> *mut c_void {
    let task: Box<T> = Box::from_raw(mem::transmute(task));
    let result = task.perform();
    mem::transmute(Box::into_raw(Box::new(result)))
}

unsafe extern "C" fn complete_task<T: BaseTask>(result: *mut c_void, out: &mut raw::Local) {
    let result: T::Output = *Box::from_raw(mem::transmute(result));
    TaskContext::with(|cx| {
        if let Ok(result) = T::complete(cx, result) {
            *out = result.to_raw();
        }
    })
}

struct PerformTask<P>(P);

pub struct TaskBuilder<'a, 'b, C, Perform, Output>
where
    C: Context<'b>,
    Perform: FnOnce() -> Output + Send + 'static,
{
    _phantom: PhantomData<&'b C>,
    context: &'a mut C,
    task: PerformTask<Perform>,
}

impl<'a, 'b, C, Perform, Output> TaskBuilder<'a, 'b, C, Perform, Output>
where
    C: Context<'b>,
    Perform: FnOnce() -> Output + Send + 'static,
{
    pub(crate) fn new(context: &'a mut C, perform: Perform) -> Self {
        Self {
            _phantom: PhantomData,
            context,
            task: PerformTask(perform),
        }
    }
}

impl<'a, 'b, C, Perform, Complete, Output> TaskBuilder<'a, 'b, C, Perform, Complete>
where
    C: Context<'b>,
    Perform: FnOnce() -> Complete + Send + 'static,
    Complete: FnOnce(TaskContext) -> JsResult<Output> + Send + 'static,
    Output: Value,
{
    pub fn schedule_task(self, callback: Handle<JsFunction>) {
        self.task.schedule(callback);
    }

    pub fn schedule(self, callback: Handle<JsFunction>) -> JsResult<'b, JsUndefined> {
        let Self { context, task, .. } = self;

        task.schedule(callback);

        Ok(context.undefined())
    }
}

impl<Perform, Complete, Output> BaseTask for PerformTask<Perform>
where
    Perform: FnOnce() -> Complete + Send + 'static,
    Complete: for<'c> FnOnce(TaskContext<'c>) -> JsResult<Output> + Send + 'static,
    Output: Value,
{
    type Output = Complete;
    type JsEvent = Output;

    fn perform(self) -> Complete {
        (self.0)()
    }

    fn complete<'a>(cx: TaskContext<'a>, complete: Complete) -> JsResult<Output> {
        (complete)(cx)
    }
}
