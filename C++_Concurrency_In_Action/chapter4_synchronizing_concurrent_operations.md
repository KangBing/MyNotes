这一章主要包括：
* 等待事件
* 等待将来的一次性事件
* 有超时事件的等待
* 使用同步操作简化代码

上一节介绍了线程之间保护共享数据，这一节讲解同步。一个线程可能需要等待另外一个线程完成某个任务后才能继续运行，这里需要的操作就是同步。C++标准库里面有支持这样的场景，使用`condition_variables`和`futures`。

### 4.1  等待一个事件或者条件
 假设一个线程要等待另外一个线程完成某个任务后才能继续执行，那么可以用一个flag来标识任务是否完成
 ```
 bool flag;
 std::mutex m;
 void wait_for_flag()
 {
 	std::unique_lock<std::mutex> lk(m);
    while(!flag)
    {
    	lk.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        lk.lock();
    }
 }
 ```
 这样的实现，等待线程消耗系统资源；检查`flag`时间未必完成合适，因为很难设置合适的sleep时间。
 
 #### 4.1.1 使用条件变量等待条件
 C++标准库有两种条件变量的实现，`std::condition_variable`和`std::condition_variable_any`，都定义在`<condition_variable>`，条件变量要和互斥量一起使用。`std::condition_variable`要和`std::mutex`一起使用；`std::condition_variable_any`可以和mutex-like类型互斥量结合使用。因为`std::condition_variable_any`更通用，它在size、performance、operating system resource有更多开销，因此应该优先选用`std::condition_variable`。
 
 下面代码实例怎么使用`std::condition_variable`来处理条件事件
 ```
 std::mutex mut;
 std::queue<data_chunk> data_queue;
 std::condition_variable data_cond;
 
 void data_prepatarion_thread()
 {
 	while(more_data_to_prepare())
	{
    	data_chunk const data = prepare_data();
        std::lock_guard<std::mutex> lk(mut);
        data_queue.push(data);
        data_cond.notify_one();
    }
 }
 
 void data_process_thread()
 {
 	while(true)
    {
    	std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(
        	lk, [] {return !data_queue.empty();});
        data_chunk data = data_queue.front();
        data_queue.pop();
        lk.unlock();
        process(data);
        if(is_last_chunk(data))
        	break;
    }
 }
 ```
使用队列在两个线程之间传递数据。
准备数据线程：当数据准备好之后，使用互斥量保护队列，随后向队列添加数据，最后通知处理数据线程已经有数据在队列。
处理数据线程：使用互斥量来保护条件变量，如果不满足条件（即队列没有数据），那么就等待，直到准备数据线程通知。
在调用`std::condition_variable.wait()`时，使用了`std::unique_lock`和lambda表达式。上面的写法可以防止*spurious wake*。
 
#### 4.1.2 使用条件变量打造一个线程安全队列
设计一个通用队列，首先应该思考要有哪些操作，具体可以参考`std::queue`。
不考虑构造函数、复制构造和赋值操作符、交换，还有三种类型操作：1、查询整个队列(`empty()`,`size()`)，2、查询队列元素(`front()`,`back()`)，3、修改队列(`push()`,`pop()`，`emplace()`)。前面讨论过接口实现问题存在的条件竞争，因此要把`front()`和`pop()`设计到一个接口中，`top()`和`pop()`一样。
 使用线程传递数据，接收数据线程一般等待数据到达，实现了连个等待接口`try_pop()`和`wait_and_pop()`。
```
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class threadsafe_queue
{
private:
	std::mutex mut;
    std::queue<T> data_queue;
    std::condition_variable data_cond;
    
public:
 	threadsafe_queue()
    {}
    threadsafe_queue(const threadsafe_queue& other)
    {
    	std::lock_guard<std::mutex> lk(other.mut);
        data_queue = other.queue;
    }
    threadsafe_queue& operator=(const threadsafe_queue&) = delete;
    
    void push(T new_value)
	{
    	std::lock_guard<std::mutex> lk(mut);
        data_queue.push(new_value);
        data_cond.notify_one();
    }
     void wait_and_pop(T& value)
     {
     	std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this]{return !data_queue.empty();});
        value = data_queue.front();
        data_queue.pop();
     }
     
     std::shared_ptr<T> wait_and_pop()
     {
     	std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this]{return !data_queue.empty();});
        std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
        data_queue.pop();
        return res;
     }
    
    bool try_pop(T& value)
    {
    	std::lock_guard<std::mutex> lk(mut);
        if(data_queue.empty())
        	return false;
         value = data_queue.front();
         data_queue.pop();
         return true;
    }
    std::shared_ptr<T> try_pop()
    {
    	std::lock_guard<std::mutex> lk(mut);
        if(data_queue.empty())
        	return std::shared_ptr<T>();
        std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
        data_queue.pop();
        return res;
    }
    

    bool empty() const
    {
    	std::lock_guard<std::mutex> lk(mut);
        return data_queue.empty();
    }
};
```
 
### 4.2 使用future等待one-off事件
有些场景，线程需要等待只发生一次的事件。标准库为这样的场景设计了`future`，`future`可以表示一次性事件，线程如果等待一次性事件，可以间隔一段时间查看`future`是否准备好，其他时间线程可以做其他事情。`future`可以关联数据，如果事件已经到达，那么`future`变为*ready*，之后`future`不能再重置。
C++标准库有两种类型的`future`，都在头文件`<future>`:*unique futures*(`std::future<>`)和*shared futures*(`std::shared_future<>`)；这样的命名和`std::unique_ptr`、`std::shared_ptr`类似。一个`std::future<>`实例只能是一个事件，而多个`std::shared_future<>`变量可能指同一个事件。使用模板就是为了关联数据。·future`是用来在线程之间同步数据的，但是它们没有同步手段，可以使用互斥量或其他方法同步。多个线程可以同时access它们自己拷贝的`std::shared_future<>`实例，而不提供数据同步方法，后面4.2.5可以看到。

#### 4.2.1 后台任务返回数值
假设有一个长期运行的后台计算任务，要取得计算的结果，`std::thread`不能直接返回结果。这时候可以用`std::async`来启动*asynchronous task*；这是不用等待线程计算结果，`std::async`返回`std::future`存储结果。取结果线程不必等待计算线程，`std::async`返回的`std::future`会携带数据，当需要去返回的数据时，在`std:;future`上调用`get()`，线程会阻塞直到取到数据。
```
#include <future>
#include <iostream>

int find_the_answer_to_ltuae();
void do_other_stuff();
int main()
{
	std::future<int> the_answer = std::async(find_the_answer_to_ltae);
    do_other_stuff();
    std::cout<<"The answer is "<<the_answer.get()<<std::endl;
    return 0;
}
```
和`std::thread`一样，`std::async`可以传入额外参数，例如启动类的成员函数要传入实例指针，还可以传入普通参数；`std::async`只能移动，不能拷贝。
```
#include <string>
#include <future>
struct X
{
	void foo(int, std::string const&);
    std::string bar(std::string const&);
};
X x;
auto f1 = std::async(&X::foo, &x, 42, "hello");
auto f2 = std::async(&X::bar, x, "goodbye");
struct Y
{
	double operator() (double);
};
Y y;
auto f3 = std::async(Y(), 3.141);
auto f4 = std::async(std::ref(y), 2.718); // 这里传入实例引用，传入指针也可以
X baz(X&);
std::async(baz, std::ref(x));
class move_only
{
public:
	move_only();
    move_only(move_only&&);
    move_only(move_only const&) = delete;
    move_only& operator = (move_only&&);
    move_only& operator = (move_only const&) = delete;
    
    void operator() ();
};
auto f5 = std::async(move_only());
```
`std::async`是否启动新线程，或等到调用同步时才执行；这取决于具体实现。可以增加一个参数来控制，`std::launch`或`std::lauch::deferred`，表面任务被延迟到调用`wait()`或`get()`才执行。`std::launch::async`指示任务在独立线程执行，`std::launch::deferred|std::launch`::async`指示根据实现决定
```
auto f6 = std::async(std::lauch::async, Y(), 1.2); //在新线程执行
auto f7 = std::async(std::launch::deferred, baz, std::ref(x)); //在调用wait()或get()时执行
auto f8 = std::async(std::launch::deferred|std::launch`::async, baz, std::ref(x)); //根据实现决定
auto f9 = std::async(baz, std::ref(x)); //根据实现决定
f7.wait(); //调用f7
```
第八章会看到使用`std::async()`来分解算法并发执行。

#### 4.2.2 把future和任务关联起来
`std::package_task<>`可以把`future`和函数/可调用对象结合起来，当调用`std::packaged_task`对象时就会调用关联的函数/可调用对象，使`future`变量变为*ready*,返回的时间存储在关联的数据结构中。
`std::packaged_task<>`模板参数是函数类型，例如`void()`表示没有参数和返回值,`int(std::string&, double*)`表示传入non-const std::string引用和double类型的指针，返回int类型。当创建`std::packaged_task<>`类型时，要传入函数或可调用对象，用来表示传入参数类型和返回类型。注意，这些参数并不需要完全匹配，编译器可能会做隐士类型转换。

`std::package_task<>`可以进行偏特化，通过成员函数`get_future()`来获取`std::future<>`，`std::future<>`关联了任务函数类型，下面是偏特化一个例子
```
template<>
class packaged_task<std::string(std::vector<char>*, int)>
{
public:
	template<typename Callable>
    explict packaged_task(Callable& f);
    std::future<std::string> get_future();
    void operator()(std::vector<char>*, int);
}
```
这样封装后，`std::package_task`就是一个可以调用对象，方便使用。例如传递到其他线程，当做参数在函数间传递，或者直接调用。当`std::package_task`封装成函数对象，传给它的参数就是异步任务的参数，结果存储在·std::future`中。

在一些GUI框架中，更新GUI要通过特定线程才行。所以要更新GUI时，可以把任务传递给更新线程即可。
```
std::mutex m;
std:;dequeue<std::package_task<void>> tasks;

bool gui_shutdown_message_received();
void get_and_process_gui_message();

void gui_thread()
{
	while(!gui_shutdown_message_received())
    {
    	get_and_process_gui_message();
        std:package_task<void> task;
        {
        	std::lock_guard<std::mutex> lk(m);
            if(tasks.empty())
            	continue;
                task = std::move(tasks.front());
                tasks.pop_front();
        }
        task();
    }
}

std::thread gui_bg_thread(gur_thread);

template<typename Func>
std::future<void> post_task_for_gui_thread(Func f)
{
	std::packaged_task<void()> task(f);
    std::future<void> res = task.get_future();
    std::lock_guard<std::mutex> lk(m);
    tasks.push_back(std::move(task));
    return res;
}
```
