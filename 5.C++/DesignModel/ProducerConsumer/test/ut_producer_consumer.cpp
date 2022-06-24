///////////////////////////////////////////////////////////
/// @copyright copyright description
/// 
/// @brief UT producer and consumer demo
/// 
/// @file ut_producer_consumer.cpp
/// 
/// @author GaoJunbao(junbaogao@foxmail.com)
/// 
/// @date 2022-06-23
///////////////////////////////////////////////////////////

// Current Cpp header
// System header
// C/C++ standard library header
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
// External library header
// Current module header
// Root directory header

namespace UTA
{

std::mutex mtx;
std::condition_variable producer, consumer;

std::queue<int> q;
constexpr int max_size = 20;

constexpr auto sleep_time = std::chrono::milliseconds(1000);

void Consumer()
{
    while (true)
    {
        std::this_thread::sleep_for(sleep_time);

        std::unique_lock<std::mutex> lock(mtx);
        while (q.size() == 0)
        {
            // 当q为空时, 消费者改为等待状态
            consumer.wait(lock); 
        }
        // 取出一个
        q.pop();
        std::cout << "consume one: " << q.size() << std::endl;
        // 只要取出一个, 表明了此时队列不满, 可以继续生产, 则可以通知生产者继续生产
        producer.notify_all();
        lock.unlock();
    }
}

void Producer(int id)
{
    while (true)
    {
        std::this_thread::sleep_for(sleep_time);

        std::unique_lock<std::mutex> lock(mtx);
        while (q.size() == max_size)
        {
            // 队列为满时, 生产者不能进行生产
            producer.wait(lock);
        }
        // 生产一个
        q.push(id);
        std::cout << "produce one: " << q.size() << std::endl;
        // 只要新生产了, 消费者即可消费
        consumer.notify_all();
        lock.unlock();
    }
}

int main()
{
    const int size = 2;
    std::thread th_producer[size];
    std::thread th_consumer[size];

    for (int i = 0; i < size; ++i)
    {
        th_producer[i] = std::thread(Producer, i + 1);
        th_consumer[i] = std::thread(Consumer);
    }

    for (int i = 0; i < size; ++i)
    {
        th_producer[i].join();
        th_consumer[i].join();
    }

    return 0;
}

} // namespace UTA

int main(int argc, char **argv)
{
    UTA::main();

    return 0;
}
