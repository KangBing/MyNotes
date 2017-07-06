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