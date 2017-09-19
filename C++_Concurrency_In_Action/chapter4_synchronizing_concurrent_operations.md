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
逻辑比较简单：跟新gui封装成一个函数，之后通过`std::package_task`封装成一个task，放到任务队列（生产者），gui线程（消费者）从队列取任务来执行。

#### 4.2.3 Making (std::)promises
如果任务比较复杂，不能使用上面的task来封装成一个函数或调用对象，那么可以使用`std::promises`。例如，当有网络连接时，如果一个连接占用一个线程，连接过多会消耗太多系统资源；一种做法是用一个（或几个线程）来处理网络连接。数据的收发都是随机的，IO线程常常在等待收/发。
`std::promise<T>`和`std::future<T>`可以提供一种方法，使用数据线程阻塞等待`std::future<T>`变为`ready`，`std::promise`可以使得数据`T`变为`ready`。通过`std::promise`的成员函数`get_future()`得到`std::future`变量，在`std::future`上调用`get_future()`阻塞等待；`std::promise`调用成员函数`set_value()`使得`std::future`变量变为`ready`。下面是IO线程处理网络IO代码。
```
#include <future>
void process_connection(connection_set& connections)
{
	while(!done(connections))
    {
    	for(connection_iterator connection = connections.begin(), end = connections.end(); connection != end(); ++connection)//遍历connections set中的每个connection
        {
        	if(connection->has_incoming_data())//有数据可以接收
            {
            	data_packet data = connection->incoming();
                std::promise<payload_type>& p = connection->get_promise(data.id);//id和std::promise关联
                p.set_value(data.payload);//之后调用std::future<data.payload>.get_future()可以获取数据data.payload
            }
            if(connection->has_outgoing_data())//有数据可以发送
            {
            	outgoing_packet data = connection->top_of_outgoing_queue();//发送队列取数据
                connection->send(data.payload);
                data.promise.set_value(true);//调用的std::future<bool>.get_future()返回true
            }
        }
    }
}
```
上面这个例子，数据收发都在IO线程。收到数据后，通过id关联的`std::promise`设置数据；发送数据时，从一个connection的发送队列取数据->发送，之后设置发送成功（std::promise.set_value(true）。
上面并没有处理异常（例如收发失败、网络断开等），主要是展示`std::promise`和`std::future`结合的用法。

#### 4.2.4 为future保存异常
`std::future`关联的函数可能抛出异常，例如：
```
double square_root(double x)
{
	if(x < 0)
    {
    	throw std::out_of_range("x < 0");
    }
    return sqrt(x);
}
std::future<double> f = std::async(square_root, -1);
double y = f.get();//此处会抛出异常
```
异常是在执行函数`square_root(double x)`函数中抛出的，但是在调用`future.get()`的线程中可以得到同样的异常（标准没有详细解释，这个异常是原始异常，或时原始异常的拷贝，不同编译器有不同实现）。同样适用于`std::packaged_task`。
在`std::promise`中提供了成员函数`set_exception()`来设置异常，可以在try/catch的catch模块设置异常:
```
extern std::promise<double> some_promise;

try
{
	some_promise.set_value(caculate_value());
}
catch(...)
{
	some_promise.set_exception(std::current_excepiton());
}
```
上面是抛出了异常；还可以选择使用`std::copy_exception()`存储新的异常，而不是抛出。
```
some_promise.set_exception(std::copy_exception(std::logic_error("foo")));
```
这样使用代码更加简洁，且应该优先这样使用；因为编译器更有可能对此做出优化。
还有一种方法在future变量中存储异常：既没有调用set函数，也没有调用packaged task，这时销毁future变量关联的`std::promise`或`std::packaged_task`。这时如果future没有*ready*，`std::promise`和`std::packaged_task`会存储`std::future_error`异常，错误代码为`std::future_errc::broken_promise`。创建promise来提供值或者异常，得到future；在没有提供值或者异常时销毁promise，如果编译器没有存储任何东西，等待线程会一直等待下去。

#### 4.2.5 在多个线程上等待
`std::future`可以在不同线程之间同步数据。如果多个线程在没有同步的情况下同时在一个`std::future`实例上等待数据，会有data rece；这是因为`std::future`的独占异步结果模型控制权和`get()`函数one-shot特性（只有一个线程可以得到数据）。
如果多个线程要等待同一个事件，可以使用`std::shared_future`。`std::future`是*moveable*，`std::shared_future`是*copyable*。因此多个实例可以关联同一个状态。在使用时，不是多个线程使用同一个`std::shared_future`变量，而是每个线程都有`std::shared_future`变量的拷贝。
`std::future`来构造`std::shared_future`时，要通过`move`剥夺其控制权
```
std::promise<int> p;
std::future<int> f(p.get_future());
assert(f.valid());
std::shared_future<int> sf(std::move(f));
assert(!f.valid());
assert(sf.valid());
```
`std::shared_future`构造函数可以为右值执行隐式转换，例如
```
std::promise<std::string> p;
std::shared_future<std::string> sf(p.get_future());
```
`std::future`还有`share()`成员函数直接转移控制权
```
std::promise< std::map< SomeIndexType, SomeDataType, SomeComparator,SomeAllocator>::iterator> p;
auto sf=p.get_future().share();
```
得到的`sf`类型为`std::shared_future< std::map< SomeIndexType, SomeDataType, SomeComparator, SomeAllocator>::iterator>`。

### 4.3 等待有限时间
线程在阻塞操作可能永远等待下去。在一些场景中，要求只是等待一段时间。总的来说有两种类型：1、等待一段时间，时间段固定；2、等待到某一个时间点。对于第一种，常用的函数后缀为`_for`，第二种为`_until`。
例如，`std::condition_variable`有两个成员函数`wait_for()`和`wait_until()`。

#### 4.3.1 时钟Clocks
Clock是时间的源信息。Clock是一个类，提供四种信息
* 当前时间*now*
* 从clock获取的时间值的类型
* 时钟的tick周期
* 时钟的tick周日是否固定，固定就是稳定的clock

当前时间可以调用静态函数`now()`获取，例如`std::chrono::system_clock::now()`返回系统时钟的当前时间。时钟的时间类型由`typedef`定义的`time_point`定义，例如`some_clock::now()`返回类型为`some_clock::time_point`。
时钟周期定义为分数，例如几分之一秒；1秒中25个周期，为`std::ratio<1, 25>`，2.5秒为一个周期定义为`std::radio<5, 2>`。如果时钟周期只能在运行时获取，或者是变化和，可以使用平均周期，或最小周期，或其他值。
如果时钟周期固定（不论是否匹配period）且不能改变，那么就说这个时钟是*steady*的。静态函数`is_steady`将会返回`true`，否则返回`false`。`std::chrono::system_cloce`不是*steady*，因为它可以调整，这会造成调用`now()`时得到的时间early。*steady*时钟很重要，`std::chrono::steady_clock`是*steady*的。`std::chrono::system_clock`表示系统的real time，可以转换为`time_t`类型；`std::chrono::high_resolution_clock`时钟周期尽可能小（分辨率高）。它们都定义在`<chrono>`中。


#### 4.3.2 Durations（时间段）
时间段用`std::chrono::duration<>`模板类表示，第一个参数是表示类型（例如`int, long, double`），第二个参数是分数，表示时间段是多少秒。例如`std::chrono::duration<shot, std::ratio<60, 1>>`表示一分钟，因为60秒是1分钟，再例如千分之一秒`std::chrono::duration<double, std::ratio<1, 1000>>`。
C++标准库在`std::chrono`用typedef定义两个很多类型的duration:nanoseconds, microseconds, milliseconds, seconds, minutes, hours；用适当的单位可以表示500年。还可以使用typedef定义的SIratios，`std::atto(10^-18)`到`std::exa(10^18)`(如果平台支持128为整数类型)。
时间段类型之间可以进行隐式类型转换，且没有数值截断（小时转换到秒可以，反之不行）。可以用`std::chrono::duration_cast<>`进行显示类型转换：
```
std::chrono::milliseconds ms(54802);
std::chrono::seconds = std::chrono::duration_cast<std::chrono::seconds>(ms);
```
结果会进行截断，而不是向上取整，这里结果是54。
时间段支持数值运算，可以进行跨类型的加减，乘除只是支持常数。例如`5 * seconds(1)`等于`seconds(5)`，也等于`minutes（1） - seconds(55)`。可以通过`count()`函数获取数值，例如`std::chrono::milliseconds(1234).count()`为1234。
基于时间段的等待，可以使用`std::chrono::duration<>`实例，例如等待future变量35milliseconds
```
std::future<int> f = std::async(some_task);
if(f.wait_for(std::chrono::millisconds(35)) == std::future_status::ready)
	do_something_with(f.get());
```
等待函数返回状态。这里等待future，因此会返回`std::future_status::timeout`或`std::future_status::ready`或`std::future_status::deferred`。基于时间段的等待函数使用的是steady时钟，35milliseconds意味着实际耗时，即时调整了时钟。当然，系统的调度和时钟精度，以及调用函数和返回等因素，常常造成等待时间多于35ms。

#### 4.3.3 Time points时间点
时钟的时间点通过模板`std::chrono::time_point<>`指定，模板第一个参数是clock，第二个参数是测量精度（`std::chrono::duration<>`实例）。时间点的数值是距离epoch时间（不同精度数值不同）。时钟可以有独立的epoch，或共享一个epoch。如果时钟共享一个epoch，那么typedef定义的`time_point`可能会有关联。可以通过`time_since_epoch()`来获取时间点，这个函数返回到epoch的时延值。
通过`std::chrono::time_point<std::chrono::system_clock, std::chrono::minutes>`返回到epochs时间的值，单位为分钟。
时间点类型`std:;chrono::time_point<>`可以和时间段进行加减运算，例如`std::chrono::high_resolution_clock::now() + std::chrono::nanoseconds(500)`将会得到500nanoseconds后的时间点。
可以通过两个时间点的减法得到时间段，例如获取代码执行时间
```
auto start = std::chrono::high_resolution_clock::now();
do_something();
auto stop = chrono::high_resulotion_clock::now();
std::cout<<"do_something() took "<<std::chrono::duration<double, std::chrono::seconds>(stop - start).cout()<<" seconds"<<std::endl;
```
`std::chrono::time_point<>`的时钟参数不仅仅指定epoch时间。例如wait函数等待参数为绝对时间时，时钟参数用来衡量时间；如果时钟周期变了，等待的时间长短会变。
等待时间点函数`_until`使用时间点方式很多，常用的一种是使用时钟偏移。系统的epoch时间time_t类型可以通过`std::chrono::system_clock::to_time_point()`获取。如果最多等待条件变量500milliseconds：
```
#include <condition_variable>
#inlcude <mutex>
#inlcude <chrono>
std::contidion_variable cv;
bool done;
std::mutex m;
bool wait_loop()
{
	auto const timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    std::unique_lock<std::mutex> lk(m);
    while(!done)
    {
    	if(cv.wait_until(lk, timeout) == std::cv_status::timeout)
        {
        	break;
        }
        return done;
    }
}
```
推荐这种方式使用条件变量等待有限时间。

#### 4.3.4 接收超时的函数
使用超时最简单的方法是给处理线程加上一个延时，当它没有处理任务时就不会占用处理器。例如在4.1节中使用的`std::this_thread::sleep_for()`和`std::this_thread::sleep_until()`，工作机制和闹钟类似：线程会sleep一段时间或到某个时间点。
sleep不是使用超时的唯一方式，还可以使用条件变量（condition variables）和future；如果锁支持超时，还可以在锁上使用，`std::mutex`和`std::recursive_mutex`不支持，但是`std::timed_mutex`和`std::recursive_timed_mutex`支持，方法为`try_lock_for()`和`try_lock_until()`。下表展示了C++标准库支持的超时函数，参数列表中表名为`duration`的必须是`std::duration<>`的实例，`time_point`的必须为`std::time_point`的实例

|Class/Namespace|Functions|Return values|
|:------------:|:--------:|:-----------:|
|std::this_thread namespace|sleep_for(duration) sleep_until(time_point)|N/A|
|std::condition_variable or std::condition_variable_any| wait_for(lock, duration) wait_until(lock, time_point)| std::cv_status::timeout or std::cv_status::no_timeout|
||wait_for(lock,duration,predicate) wait_until(lock,time_point,predicate)|bool-the return value of predicate when awakened|
|std::time_mutex or std::recursive_timed_mutex|try_lock_for(duration) try_lock_until(time_point)|bool-true if lock was acquired, false otherwise|
|std::unique_lock<TimedLockable|unique_lock(lockable, duration) unique_lock(lockable, time_point)|bool-true if the lock was acquired, false otherwise|
|std::future<ValueType> or std::shared_future<ValueType>|wait_for(duration) wait_until(time_point)|std::fature_status::timeout if wait timed out, std::future_status::ready if the future is ready, or std::future_status::defered is the future holds a deferred function the han't yet started|


### 4.4 使用同步操作简化代码