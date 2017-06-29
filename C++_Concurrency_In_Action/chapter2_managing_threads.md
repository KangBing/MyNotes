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

#### 2.1.4 在后台运行线程
调用`detach()`可以使得`std::thread`对象在后台运行，不能再用直接方式跟它通信，不能再通过`std::thread`对象引用它；线程的所有权和控制权交给了C++ Runtime Library，有运行库来确保线程退出时回收资源。
UNIX的的后台进程常常叫做守护进程，分离的线程叫做守护线程。守护线程一般是长时间运行（几乎和应用生命周期相同），执行后台任务：检测文件系统、清理不用资源、优化数据结构等。可以用后台线程监控其他线程是否完成，线程在哪里执行"fire and forget"任务。
`std::thread`对象调用`detach()`后，就不再和线程有关联了，也不能再调用`joinable`了。
```
std:thread t(do_background_work);
t.detach();
assert(!t.joinable());
```

### 2.2 向线程函数传递参数
向线程运行的可调用对象或者函数传递参数，就像给`std::thread`构造函数传递参数一样。记住：默认这些参数*拷贝*到线程空间，即使参数是引用。下面是一个例子：
```
void f(int i, std::string const& s);
std::thread t(f, 3, "hello");
```
传入的类型时`char const*`，在新线程的上下文中转换为`std::string`。当传入的指针指向动态变量时，需要注意：
```
void f(int i, std::string const& s);
void oops(int some_param)
{
	char buffer[1024];
    sprintf(buffer, "%i", some_param);
    std::thread t(f, 3, buffer);
    t.detach();
}
```
传入参数指向函数`oops`内部变量，当`oops`早于线程执行完时，内部变量`buffer`释放，将会造成未定义行为。解决方法为显式转换为`std::string`
```
void f(int i, std::string const& s);
void oops(int some_param)
{
	char buffer[1024];
    sprintf(buffer, "%i", some_param);
    std::thread t(f, 3, std::string(buffer));
    t.detach();
}
```

还有一种相反的场景：对象拷贝到线程了，但是你却想要拷贝引用。例如要在新的线程更新数据，之后使用
```
void update_data_for_widget(widget_id w, widget_data& data);

void oops_again(widget_id w)
{
	widget_data data;
    std::thread t(update_data_for_widget, w, data);
    display_status();
    t.join();
    process_widget_data(data);
}
```
x希望`update_data_for_widget`第二个参数是引用，但实际是拷贝。如果确实要传入引用，使用`std::ref`
```
std::thread t(update_data_for_widget, w, std::ref(data));
```

`std::bind`和`std::thread`参数传递机制一样。传递类的成员函数，第一个参数为`this`指针
```
class X
{
public:
	void do_lengthy_work();
};
X my_x;
std::thread t(&X::do_lengthy_work, &my_x);
```
 还有一种场景，参数不能`copy`，只能`move`，例如`unique_ptr`  。
 ```
 void process _big_object(std::unique_ptr<big_object>);
 std::unique_ptr<big_object> p(new big_object);
 p->prepare_data(42);
 std::thread t(process_big_object, std::move(p));
 ```
### 2.3 转移线程所有权
创建一个线程，需要在另外一个线程等待这个线程运行结束；这是就需要把这个线程对象传入到另外那个线程。
`std::thread`时resource-owning类型，和`std::ifstream`、`std::unique_ptr`，是*moveable*不是*copyable*。
```
void some_function();
void some_other_function();
std::thread t1(some_function);
std::thread t2 = std::move(t1); // 调用右值赋值构造函数
t1 = std::thread(some_other_function); // 临时对象赋值给t1,不用显式调用std::move
std::thread t3;
t3 = std::move(t2); // 调用右值赋值操作符
t1 = std::move(t1);
```

move支持把`std::thread`对象传出函数作用域
```
std::thread f()
{
	void some_function();
    return std::thread(some_function);
}
std::thread g()
{
	void some_other_function(int);
    std::thread t(some_other_function, 42);
    return t;
}
```

把`std::thread`传入函数时，可以构造临时对象传入
```
void f(std::thread t);
void g()
{
	void some_function();
    f(std::thread(some_function));
    std::thread t(some_function);
    f(std::move(t));
    
}
```

可以通过move以及RAII构造scoped_thread，确保线程j对象析构前join
```
class scoped_thread
{
	std::thread t;
public: 
	explicit scoped_thread(std::thread t_):
    	t(std::move(t_))
    {
    	if(!t.joinable())
        	throw std::logic_error("No thread");
    }
    ~scoped_thread()
    {
    	t.join();
    }
	scoped_thread(scoped_thread const&)=delete;
    scoped_thread& operator=(scoped_thread const&)=delete;
};
```

move还支持容器，可以把线程放到容器
```
void do_word(unsigned id);
void f()
{
	std::vector<std::thread> threads;
    for(unsigned i = 0; i < 20; ++i)
    {
    	threads.push_back(std::thread(do_work, i));
    }
    std::for_each(threadsbegin(), threads.end(),
    	std::mem_fn(&std::thread::join));
}
```

### 2.4 选择合适数量的线程
`std::thread::hardware_concurrency()`返回C++ Standard Library可以同时并发运行的线程数，显然多核机器为核数。这只是一个提示，如果不能得到正确信息将会返回0。
下面是并发版本的`std::accumulate`
```
template<typename Iterator, typename T>
struct accumulate_block
{
	void opeartor() (Iterator first, Iterator last, T& result)
    {
    	result = std::accumulate(first, last, result);
    }
};
template<typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init)
{
	unsigned long const length = std::distance(first, last);
    if(!length)
    	return init;
        
    unsigned long const min_per_thread = 25;
    unsigned long const max_thread = (length + min_per_thread - 1) / min_per_thread;
    
    unsigned long const hardware_threads = std::thread::hardware_concurrency();
    
    unsigned long const num_threads = std::min(hardware_threads != 0? hardware_threads: 2, max_threads);
    
    unsigned long const block_size = length / num_threads;
    
    std::vector<T> result(num_threads);
    std::vector<std::thread> threads(num_threads-1);
    
    Iterator block_start = first;
    for(unsigned long i = 0; i < (num_threas - 1); ++i)
    {
    	Iterator block_end = block_start;
        std::advance(block_end, block_size);
        threads[i] = std::(accumulate_block<Iterator, T)(), block_start, block_end, std::ref(results[i]));
        block_start = block_end;
    }
    accumulate_block<Iterator, T>()(block_start, last, results[num_threads - 1]);
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
    result std::accumulate(results.begin(), results.end(), init);
}
```
上面带来流程比较简单，就是把一个范围(first, last)内数据，根据线程数划分，不同线程accumulate不同区间，最终再合并这些区间accumulate的结果。当然要注意一些边界和特殊情况的处理。

### 2.5 识别线程
两种方式可以获取线程id；一种是通过线程`std::thread`对象调用`get_id()`，另一种是在这个线程运行代码中调用`std::this_thread::get_id()`。前一种，如果`std::thread`对象已经没有关联线程了，那么也会返回一个id，这样的id意味着"not any thread"。
`std::thread::id`对象可以拷贝、比较。两个id相同，要么是就是同一个线程（包括都是not any thread)；不同则不是同一个线程（或者其中一个是not any thread)。`std::thread::id`实现了很多比较操作，可以作为关联容器的key，可以排序;如果a < b, b < c，那么a < c。标准库也提供了`std::hash<std::thread::id>`，所以也可以放到无序容器中。
可以根据id让不同线程，执行不同任务
```
std::thread::id master_thread;
void some_core_part_of_algorithm()
{
	if(std::this_thread::get_id() == master_thread)
    {
    	do_msater_thread_work();
    }
    else
    {
    	do_common_work();
    }
}
```

### 2.6 总结
这一种介绍了启动线程、等待线程结束、分离线程做后台线程；向线程传递参数；转移线程控制权；区分线程。



