## Sharing data between threads
同一个进程之间的线程共享数据方式比较简单；共享数据时要确保数据一致，避免因为竞争引起的问题。

### 3.1 线程之间共享数据的问题
线程之间共享数据问题的根源是修改数据。如果所有的共享数据都是*read-only*，就不存在共享数据的问题了。

#### 3.1.1 条件竞争
条件竞争，是指在多线程并发执行中；结果依赖线程的执行顺序。有时候结果是良性的，因为我们可以接受任何一个结果。当讨论条件竞争时，常常是指有问题的条件竞争，不是指良性条件竞争。C++标准定义了条件竞争：多个线程修改同一个对象，引起未定义行为。

#### 3.1.2 避免有问题的条件竞争
最简单的办法就是提供一种保护机制，同一时刻只有一个线程可以修改数据，可以看到修改过程中的中间状态；对于其他线程，要么修改还没卡开始，要么修改已经完成。

还有一种就是修改数据结构的设计，修改是一系列不可拆分的变化。这也是通常说的*无锁编程*，比较难得到正确结果。在这一级别编程，内存模型的细微差别，不同线程访问不同数据，都将变得更加复杂。

另外一种方式是把修改当做*事务*。这点类似数据库的概念。

C++标准库提供的保护共享数据最基本的机制是*mutex*。

### 3.2 使用互斥量保护共享数据
保护共享数据，确保访问共享数据的代码互斥执行即可。使用互斥量可以实现保护共享数据。
互斥量是C++中保护共享数据常用的机制。使用互斥量时1、确保保护的临界区合适2、防止死锁。

#### 3.2.1 在C++中使用互斥量
`std::mutex`可以创建互斥量对象，`lock()`给互斥量上锁，`unlock()`给互斥量解锁。为了防止给互斥量解锁，常常使用RAII机制，`std::lock_guard`模板来上锁解锁。下面是使用互斥量保护list的例子
```
#include <list>
#include <mutex>
#include <algorithm>

std::list<ing> some_list;
std::mutex some_mutex;

void add_to_list(int new_value)
{
	std::lock_guard<std:mutex> guard(some_mutex);
    some_list.push_back(new_value);
}
bool list_contains(int value_to_find)
{
	std::lock_guard<std::mutex> guard(some_mutex);
    return std::find(some_list.begin(), some_list.end(), value_to_find) != some_list.end();
}
```

#### 3.2.2 设计保护数据的代码
保护共享数据并不像上面那样在函数中加上一个`std::lock_guard`那么简单；如果得到共享数据的引用或指针，上面的保护也变得没有意义。例如，函数返回了共享数据的引用或指针
```
class some_data
{
	int a;
    std::string b;
public:
	void do_something();
};

class data_wrapper
{
private:
	some_data data;
    std::mutex m;
public:
	template<typename Function>
    void process_data(Function func)
    {
    	std::lock_guard<std::mutex> l(m);
        func(data); // 共享数据传递给用户自定义函数
    }
};

some_data* unprotected;
void malicious_function(some_data& protected_data)
{
	unprotected = &protected_data;
}
data_wrapper x;

void foo()
{
	x.process_data(malicious_function);
    unprotected->do_something(); // 没有保护使用共享数据
}
```
上面例子中看似使用了互斥量保护共享数据，但是在用户自定义函数中，将共享数据以指针形式传递出来；之后再使用这个指针式时，并没有使用互斥量保护。
不要把共享数据的指针或引用传递到互斥量保护的区域以外。

#### 3.2.3 找出接口中的条件竞争
使用互斥量或其他机制避免条件竞争，要确定保护了共享数据。
以`std::stack`为例，除了`swap()`，还有5个接口`push(), pop(), top(), size(), empty()`。如果修改接口，返回copy，而不是返回reference，使用互斥量保护内部数据，这个接口还是存在条件竞争。不仅基于互斥量的实现有这个问题，无锁实现也存在这个问题。
```
template<typename T, typename Container = std::dequeue<T> >
class stack
{
public:
	explicit stack(const Container&);
    explicit stack(Container&& == Container());
    template <class Alloc> stack(const  Container&, const Alloc&);
    template <class Alloc> stack(Container&&, const Alloc&);
    template <class Alloc> stack(Container&&, const Alloc&);
    template <class Alloc> stack(stack&&, const Alloc&);
    
    bool empty() const;
    size_t size() const;
    T& top();
    T const& top() const;
    void push(T const&);
    void push(T&&);
    void pop();
    void swap(stack&&);
}
```
接口·empty(), size()`的接口并不可靠，因为调用时得到的结果，再使用时可能stack有元素push/pop了。
```
stack<int> s;
if(!s.empty())
{
	int const value = s.top();
    s.pop();
    do_something(value);
}
```
上面代码单线程执行，没有问题。如果多线程，就有危险。在空的stack上调用`top()`会有未定义行为。stack内部使用互斥量并不能解决上面问题，这是接口设计的问题。
最简单的改进方法是改进`top()`，当栈空时调用`top()`抛出异常；但是这样`empty()`将会变得多余，且在调用时`top()`时要捕捉异常。
一个比较激进的改进是把`top()`和`pop()`结合起来，用互斥量来保护。Tom Cargill指出，这样的联合调用会有新的问题，例如拷贝构造函数的对象在栈上会抛出异常（可能是栈空间不够用？）。Herb Sutter可以从异常安全角度完美解决这个问题，但是潜在的条件竞争会有新的组合。

例如`stack<vector<int>>`， `vector`是动态分配的容器，当拷贝`vector`时，需要动态分配内存拷贝元素；这时可能抛出'std::bad_alloc`异常。当`pop()`函数返回出栈值时，如果元素已经出栈，但是在返回时抛出异常；这样会造成栈内数据丢失。因此`top()`和`pop()`设计成2个接口。
接口的分割造成了条件竞争，但是还有其他选择，这些选择会有额外代价。

* Option 1：传递引用
传递一个引用，来存储返回的值
```
std::vector<int> result;
some_stack.pop(result);
```
许多时候可以这样做；但是它需要提前构造堆栈存储元素的实例。对于一些特别花时间/资源的类型，这样实践不可行；在构造时也未必有构造函数需要的参数。当然，存储要素类型要支持赋值操作符。

* Option 2: 要求无异常的拷贝构造函数或移动构造函数。
如果`pop()`返回值，只是存在一个异常安全的问题。许多类型的拷贝构造函数不抛出异常，一些传递右值的移动赋值构造函数也不抛出异常。一种做法是严格限制线程安全栈的使用，使得值可以安全返回没有异常。
这样虽然安全，但不切实际。编译时使用`std::is_nothrow_copy_constructible`或`std::is_nothrow_move_constructible`来检测拷贝/移动构造函数不抛出异常，但是限制太多。用户自己定义的类型，拷贝构造函数会抛出异常，或者没有移动构造函数。这样的类型，不能存储到线程安全栈。

* Option 3: 返回指向栈顶元素的指针
返回指向栈顶元素的指针可以避免异常安全的问题。但是有两个缺点1、需要管理内存，这一点可以使用智能指针`std::shared_ptr`解决，2、对于简单类型，例如`int`有额外开销。这样栈中对象都需要new出来，相对非线程安全版本，开销比较大。

* Option 4: Option 1+2或Option 1+3
通用性代码中，接口灵活性不能忽视。如果选择了Option2或Option 3，那么实现Option 1也不难。

线程安全栈的例子，实现了Option 1和Option 3，`pop()`有两个版本
```
#include <exception>
#include <memory>

struct empty_stack: std::exception
{
	const char* what() const throw();
};

template<typename T>
class threadsafe_stack
{
private:
	std::stack<T> data;
    mutable std::mutex m;
public:
	threadsafe_stack() {}
    threadsafe_stack(const threadsafe_stack& other)
    {
    	std::lock_guard<std::mutex> lock(other.m);
        date = other.data;
    }
    threadsafe_stack& operator=(const threadsafe_stack&) = delete;
    
    void push(T new_value)
    {
    	std::lock_guard<std::mutex> lock(m);
        data.push(new_value);
    }
    std::shared_ptr<T> pop()
    {
    	std::lock_guard<std::mutex> lock(m);
        if(data.empty()) throw empty_stack();
        std::shared_ptr<T> const res(std::make_shared<T>(data.top()));
        data.pop();
        return res;
    }
    void pop(T& value)
    {
    	std::lock_guard<std::mutex> lock(m);
        if(data.empty()) throw empty_stack();
        value = data.top();
        data.opo();
    }
    bool empty() const
    {
    	std::lock_guard<std::mutex> lock(m);
        return data.empty();
    }
};
```

锁的粒度太小，不能完全覆盖需要保护的部分；锁的粒度太大，容易导致性能下降。如果使用多个锁，那么可能会有死锁的问题。

#### 3.2.4 死锁：问题及方案
如果一个操作可能需要锁住多个互斥量，就可能造成死锁。
常见的一个解决方案是，在使用多个锁时，按照固定顺序来上锁，例如使用mutex A和mutex B时，都先给A上锁，再给B上锁。这样操作，有些不易实现，例如交换两个变量，每个变量内部都有锁。
C++标准库中有解决办法，使用`std:;lock`，可以同时给多个互斥量上锁，不会发生死锁。
```
class some_big_object;
void swap(some_big_object& lhs, some_big_object& rhs);

class X
{
private:
	some_big_object some_detail;
    std::mutex m;
public:
	X(some_big_object const& sd):some_detail(sd){}
    friend void swap(X& lhs, X& rhs)
    {
    	if(&lis == & rhs)
        	return;
        std::lock(lhs.m, ths.m); // 同时上锁
        std::lock_guard<std::mutex> lock_a(lhs.m, std::adopt_lock); // adopt_lock表示已经上锁了
        std::lock_guard<std::mutex> lock_b(rhs.m, std::adopt_lock);
        swap(lhs.some_detail, ths.some_detail);
    }
};
```
`std::lock`可能会抛出异常，但是它会确保要么都上锁成功，要么都没有上锁。`std::lock`可以在某些场景下帮助避免死锁；解决死锁的问题，还要依赖开发者经验水平。

### 3.2.5 避免死锁进一步指导
大多情况下的死锁都发生在使用锁的时候，不使用锁时也可能发生死锁，例如两个线程互相调用对方线程对象的`join()`。
避免死锁的一个基本原则可以归纳为：不要等待一个可能等待你的线程。

* 避免嵌套锁
当持有一个锁时，不要再去获取另外的锁。如果需要同时获取多个锁，可以使用`std::lock`来同时获取。

* 持有锁时不要调用用户自定义代码
这个原则是上条原则的补充。持有锁时，再调用自定义代码，有危险。因为不知道自定义代码是否又去获取锁。但有时候难以避免这种情况，这是有新的指导原则。

* 以特定顺序获取锁
如果需要获取多个锁，又不能使用`std::lock`，那么就以特定顺序来获取锁。实践中，可以根据锁的地址来给锁排序，以此来决定上锁顺序。

* 分层次使用锁
这是比较特殊的情况，可以在运行时检查是否可以使用锁。例如应用分为多层，在加锁时，要先给高层加锁，再给底层加锁。
```
hierarchical_mutex high_level_mutex(10000);
hierarchical_mutex low_level_mutex(5000);

int do_low_level_stuff();

int low_level_func()
{
	std::lock_guard<hierarchical_mutex> lk(low_level_mutex);
    return do_low_level_stuff();
}

void high_level_stuff(int some_param)

void high_level_func()
{
	std:;lock_guard<hierarchical_mutex> lk(high_level_mutex);
    high_level_stuff(low_level_func());
}

void thread_a()
{
	high_level_func();
}

hierarchical_mutex other_mutex(100);
void do_other_stff();

void other_stuff()
{
	high_level_func();
    do_other_stuff();
}

void thread_b()
{
	std::lock_guard<hirarchical_mutex> lk(other_mutex);
    other_stuff();
}
```
`thread_a()`没有问题，因为它先给高层互斥量加锁，再给底层互斥量加锁。`thread_b()`有问题，加锁顺序不对。
`hierarchical_mutex`用户可以自己实现。下面是一个简单的实现，它可以使用`std::lock_guard<>`，因为它实现了`lock()`,`unlock()`,`try_lock()`。
```
class hierarchical_mutex
{
	std::mutex internal_mutex;
    unsigned long const hierarchy_value;
    unsigned long previous_hierarchy_value;
    static thread_local unsigned long this_thread_hierarchy_value;
    
    void check_for_hierarchy_violation()
    {
    	if(this_thread_hierarchy_value<= hierarchy_value)
        {
        	throw std::logic_error("mutex hierarchy violated");
        }
    }
    void update_hierarchy_value()
    {
    	previous_hierarchy_value = this_thread_hierarchy_value;
        this_thread_hierar_value = hierarchy_value;
    }
   
public:
	explicit hierarchical_mutex(unsigned long value): hierarchy_value(value), previous_hierarch_value(0)
    {}
    void lock()
    {
    	check_for_hierarcy_violation();
        internal_mutex.lock();
        update_hierarchy_value();
    }
    void unlock()
    {
    	this_thread_hierarchy_value = previous_hierarchy_value;
        internal_mutex.unlock();
    }
    bool try_lock()
    {
    	check_for_hierarchy_violation();
        if(!internal_mutex.try_lock())
        {
        	return false;
        }
        update_hierarchy_value();
        return true;
    }
};
thread_local unsigned long hierarchical_mutex::this_thread_hierarchy_value(ULONG_MAX);
```

* 把这些准则扩展到锁以外
死锁不仅仅发生在使用时的时候，任何循环等待的同步操作都可能发生死锁。可以把上面准则扩展到锁以外使用。

#### 3.2.6 使用`std::unique_lock`灵活上锁
与`std::lock_guard`相比，`std::unique_lock`更加灵活。`std::unique_lock`不一定时刻持有关联的互斥量，例如构造时传入第二个参数`std::adopt_lock`表示只是管理互斥量上的锁；传入第二个参数`std::defer_lock`表示在构造时，互斥量应该处于解锁状态。
更多内容可以参考[这里](http://www.cplusplus.com/reference/mutex/unique_lock/)

#### 3.2.7 在不同作用域转移互斥量所有权
`std::unique_lock`实例不一定持有关联的互斥量，所以互斥量的所有权可以通过*moving*方式在不同实例间转移。在一些场景中自动转移，例如函数返回实例；在一些场景中需要显式调用`std::move()`。根本上来说，要看源实例是左值还是右值。
`std::unique_lock`可以*movable*，但是不能*copyable*。下面是一个例子
```
std::unique_lock<std::mutex> get_lock()
{
	extern std::mutex some_mutex;
    std::unique_lock<std::mutex> lk(some_mutex);
    prepare_data();
    return lk; // 直接返回，无需std::move
}
void process_data()
{
	std::unique_lock<std::mutex> lk(get_lock());
    do_something();
}
```

#### 3.2.8 使用合适粒度的锁
锁的粒度是指锁包含的范围。细粒度的锁包含小范围的数据，粗粒度的锁包含大范围的数据。
在可能的情况下，只有在access共享数据时才加锁，在锁的外面再处理数据。特别是，在持有锁时不要进行耗时操作（例如文件I/O）。
`std::unique_lock`可以调用`unlock`解锁，之后可以再调用`lock`上锁
```
void get_and_process_data()
{
	std::unique_lock<std::mutex> my_lock(the_mutex);
    some_class data_to_process = get_next_data_chunk();
    my_lock.unlock();
    result_type result = process(data_to_process);
    my_lock.lock();
    write_result(data_to_process, result);
}
```
以合适粒度加锁，不仅仅是关于数据量，还包括持有锁的时间以及持有锁时进行的操作。通常来说，在需要锁保护的操作上，持有锁的时间应该尽量的少。
在前面的例子中，交换两个类实例时需要比较它们是否相同。比较简单数据类型时（int)，简单数据类型拷贝代价非常小，可以先拷贝再比较。
```
class Y
{
private:
	int some_detail;
    mutable std::mutex m;
    
    int get_detail() const
    {
    	std::lock_guard<std::mutex> lock_a(m);
        return some_detail;
    }
public:
	Y(int sd):some_detail(sd){}
    
    friend bool operator==(Y const& lhs, Y const& rhs)
    {
    	if(&lhs == &rhs)
        	return true;
        int const lhs_value = lhs.get_detail();
        int const rhs_value = rhs.get_detail();
        return lhs_value == rhs_value;
    }
};
```
上面代码中，在比较前先获取值，比较时没有锁，因此比较的那一时刻会有条件竞争，比较的返回值并不完全“正确”。


### 3.3 保护共享数据的其他方法
尽管互斥量是最常用的机制，但是在特殊场景中还可以使用其他方法来保护共享数据。
例如，共享数据只有在并发情况下初始化时才需要保护（可能初始化后是只读的）。这种情景，每次access都加锁代价有点大。

#### 3.3.1 初始化时保护共享数据
数据初始化代价比较大，例如是连接数据库，分配内存等。使用*lazy initialization*，单线程做法通常是检测是否初始化，如果没有则初始化：
```
std:::shared_ptr<some_resource> resource_ptr;
void foo()
{
	if(!resource_ptr)
    {
    	resource_ptr.reset(new some_resource);
    }
    resource_ptr->do_somethring();
}
```
当使用多线程时，直接把上面代码“翻译”，每个线程在检查和初始化时，都加锁
```
td:::shared_ptr<some_resource> resource_ptr;
std::mutex resource_mutex;
void foo()
{
	std::unique_ptr<std::mutex> lk(resource_mutex);
    if(!resource_ptr)
    {
    	resource_ptr.reset(new some_resource);
    }
    lk.unlock();
    resource_ptr->do_somethring();
}
```
上面代码可以优化，使用“Double-Checked Locking"
```
void undefined_behaviour_with_double_checked_locking()
{
	if(!resource_ptr) // 1
    {
    	std::lock_guard<std::mutex> lk(resource_mutex);
        if(!resource_ptr) //2 
        {
        	resource_ptr.reset(new some_resource); //3
        }
    }
    resource_ptr->do_something();//4
}
```
上面代码有数据竞争。1处判断时并没有使用锁，它会和3处有数据竞争，不仅仅涉及指针，还涉及指针指向对象。因为new操作创建对象分两步，首先分配内存，之后再分配内存位置调用构造函数。可能在分配内存，且为构造对象是调用4处，会出现未定义行为。
C++标准委员会为解决这样场景给出了新的方法，即使用`std::call_once`和`std::once_flag`。每个线程都调用`std::call_once`来初始化，在`std::call_once`返回时就能正确初始化了；这样代价比使用互斥量小。`std::call_once`可以和任何可调用对象结合使用
```
std::shared_ptr<some_resource> resource_ptr;
std::once_flag resource_flag;

void init_resource()
{
	resource_ptr.reset(new some_resource);
}

void foo()
{
	std::call_once(resource_flag, init_resource);
    resource_ptr->do_something();
}
```

·std::once_flag`还可以用作延迟初始化类成员
```
class X
{
private:
	connection_info connection_details;
    connection_handle connection;
    std::once_flag connection_init_flag;
    
    void open_connection()
    {
    	connection = connection_manager.open(connection_detail);
    }

public:
	X(connection_info const& connection_details_):
    	connection_details(connection_details_)
    {}
    void send_data(data_packet const& data)
    {
    	std::call_once(connection_init_flag, &x::open_connection, this);
    }
    data_packet receive_data()
    {
    	std::call_once(connection_init_flag, &X::open_connection, this);
        return connection.receive_data();
    }
};
```
上面代码中，要么`send_data()`初始化连接，要么`receive_data()·初始化连接。需要注意的是`std::once_flag`既不能copied，又不能moved。

多线程初始化局部`static`变量时，在C++11之前不是线程安全的，有了C++11之后，是线程安全的：
```
class my_class;
my_class& get_my_class_instance()
{
	static my_class instance;
    return instance;
}
```

#### 3.3.2 保护很少更新的数据结构
有些场景中，数据更新很少，大部分操作都是读取数据。例如DNS缓存，很少会去更新。这样的场景，在使用数据（读/写）是都加锁代价过大，因为大部分操作都是读数据，并不需要加锁。这是需要的是`reader-writer`互斥量：写线程独占数据，读线程可以并发访问数据。
C++标准库没有提供这样的互斥量。可以使用boost库中的`boost::shared_mutex`，在写线程中使用`std::lock_guard<boost::shared_mutex>`或`std::unique_lock<boost::shared_mutex>`，在读线程中使用`boost::shared_lock<boost::shared_mutex>`。线程如果想要独占`boost::shared_lock`就会阻塞直到其他线程释放了`boost::shared_lock`。
```
#include <map>
#include <string>
#include <mutex>
#include <boost/thread/shared_mutex.hpp>

class dns_entry;

class dns_cache
{
	std::map<std::string, dns_entry> entries;
    mutable boost::shared_mutex entry_mutex;
public:
	dns_entry find_entry(std::string const& domain) const
    {
    	boost::shared_lock<boost::shared_mutex> lk(entry_mutex);
        std::map<std::string, dns_entry>::const_iterator const it = entries.find(domain);
        return (it == entries.end())? dns_entry: it->second;
    }
    void update_or_add_entry(std::string const& domain, dns_entry const& dns_details)
    {
    	std::lock_guard<boost::shared_mutex> lk(entry_mutex);
        entries[domain] = dns_detail;
    }
};
```

#### 3.3.3 递归锁
使用`std::mutex`时，如果一个线程已经给它加锁，再次加锁时会导致未定义行为。但是在一些场景用，需要一个线程给互斥量多次加锁。C++标准委提供了这样的锁`std::recursive_mutex`。一个线程可以多次给`std::recursive_mutex`加锁，但要确保加锁和解锁次数相同，可以使用RAII来确保。
大多时候，如果你需要使用递归锁，你要认真思考你的代码结果，尽量避免使用它。例如一个类的成员函数都加锁，一个成员函数调用另一个成员函数，这种场景可以封装为一个函数，在一个函数中调用这两个成员函数。

### 3.4 总结
这一章讲解了数据竞争，使用互斥量保护数据，死锁以及解决方法，变量初始化，保护读多写少的数据，递归锁。