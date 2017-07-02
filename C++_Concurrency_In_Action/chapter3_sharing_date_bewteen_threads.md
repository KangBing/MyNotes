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