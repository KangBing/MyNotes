现在基本管理包括
* 启动新线程，运行特定代码
* 等待线程结束，分离线程
* 线程唯一标识

### 2.1 基本线程管理
一个程序至少包含一个线程。线程运行的函数返回，线程就结束。

#### 2.1.1 启动线程
创建一个`std::thread`对象就启动了一个新的线程，例如
```
void do_some_work();
std::thread my_thread(do_some_work);
```

`std:thread`创建时不一定传入函数，传入可调用类型（callable type）都可以，例如：
```
class background_task
{
public:
	void opeartor()() const
    {
    	do_something();
        do_something_else();
    }
}
backgroupd_task f;
std:thread my_thread(f);
```
上面代码中，函数对象拷贝到新的线程存储空间，要确保拷贝对象和原对象相同。

传入函数对象时，要避免语法歧义，例如
```
std::thread my_thread(backgroud_task());
```
这里会把`my_thread`当做函数，返回类型为`std::thread`，参数为函数指针（返回backgroud_task的函数指针）。可以通过下面方式避免歧义：
```
std::thread my_thread((background_task())); // 多加一个括号
std::thread my_thread{background_task()};// 新的统一初始化方式
```

可调用类型可以使`lambda expression`；
```
std::thread my_thread([]{
	do_something();
    do_something_else();
});
```

创建线程后，要显式决定是否等待它结束（通过join），或让它自己运行（通过detach）。`std::thread`析构时会调用`std::terminate`，终止线程；因此即使可能存在异常，也要确保线程join或detach。要确保在线程对象`std::thread`析构前join或detach；如果线程已经终止，如果再detach，那么线程可能在`std::thread`析构后继续运行。

如果detach线程，要确保线程使用的数据一直有效。例如线程可能会使用局部对象或者指针，要防止这些数据失效。防止这个问题的一个办法是拷贝数据到线程内；还有一个方法是等待线程运行结束，再销毁线程使用的共享数据。

#### 2.1.2 等待线程结束
通过调用`std::thread.join()`就可以等待线程结束。实际中很少这么做，不会拿出线程单独等待其他线程结束。
`join`简单粗暴-等待或者不等待。如果需要更优雅的方式控制线程，例如检查线程是否结束，等待一段时间，那么可以使用其他机器例如条件变量或`futures`。

调用`join`后，还会清理线程存储等，此后std::thread对象和线程再无关联，也就是说只可以条用一次`join`，之后再调用`joinable()`就会返回`false`。

#### 2.1.3 异常环境中等待
前面已经提到过，要在`std::thread`对象析构前`join()`或`detach()`线程。如果分离，在创建`std::thread`对象后离开调用`detach()`，这没有问题。但是如果调用`join()`，要选择在什么位置调用`join()·，因为可能因为异常跳过某些代码。

在存在异常的环境
```
struct func;
void f()
{
	int some_local_state = 0;
    func my_func(some_local_state);
    std::thread t(my_func);
    try
    {
    	do_	somethring_in_current_thread();
	}
    catch(...)
    {
    	t.join();
        throw;
    }
    t.join();
  }
```
还有一种方式RAII
```
class thread_guard
{
	std::thread& t;
public:
	explicit thread_guard(std`::thread& t_):t(t_{}
    ~thread_guard()
    {
    	if(t.joinable())
        {
        	t.join();
        }
    }
    thread _guard( thread_guard const&)=delete;
    thread_guard& operator=(thread_guard const&) = delete;
    
}

void f()
{
	int some_local_state = 0;
    func my_func(some_local_state;
    std::threadt(my_func);
    thread_guard g(t);
    do_something_in_current_thread();
}
```

当调用到函数`f`末尾时，会析构局部对象，析构时会调用`join()`。
