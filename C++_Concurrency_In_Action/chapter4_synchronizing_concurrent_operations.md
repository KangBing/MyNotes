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
 不考虑构造函数、复制构造和赋值操作符、交换，还有三种类型操作：1、查询整个队列(`empty()`,`size()`)，2、查询队列元素(`front()`，`back()`)，3、修改队列（`push()`,`pop()，`emplace()`）。前面讨论过接口实现问题存在的条件竞争，因此要把`front()`和`pop()`设计到一个接口中，`top()`和`pop()`一样。
 使用线程传递数据，接收数据线程一般等待数据到达，实现了连个等待接口`try_pop()`和`wait_and_pop()`。
 ```
 #include <memory>
 
 template <typename T>
 class threadsafe_queue
 {
 	threadsafe_queue();
    threadsafe_queue(const threadsafe_queue&);
    threadsafe_queue& operator=(const threadsafe_queue&) = delete;
    
    void push(T new_value);
    
    bool try_pop(T& value);
    std::shared_ptr<T> try_pop();
    
    void wait_and_pop(T& value);
    std::shared_ptr<T> wait_and_pop();
    
    bool empty() const;
 };
 ```
 
 
 
 
 
 
 
 
 
 
 
 
 
 
