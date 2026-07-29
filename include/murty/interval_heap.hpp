/**
 * @file      interval_heap.hpp
 * @brief     Implemts interval heap
 * @author    @vojtagart
 * @date      21/02/2026
 * @see       https://github.com/Vojtagart/murty
 * * @par Credits & Acknowledgments
 * - Algorithm inspired by: Informatica, Vakgreep & H, Budapestlean & Van Leeuwen, Jan & Leeuwen, Jan & Wood, Derick. (2001). Interval Heaps. 
 * - Documentation formatted and refined with the assistance of Google Gemini
 * * @copyright Copyright (c) 2026 @vojtagart. Released under the MIT License.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <functional>
#include <cassert>
#include <type_traits>


namespace murty {

/**
 * @brief Interval heap implementation
 * 
 * Efficient implementation of interval binary heap
 * 
 * @tparam T - the type of the stored elements
 * @tparam Compare - a type providing a strict weak ordering
 */
template <typename T, class Container=std::vector<T>, class Compare=std::less<typename Container::value_type>>
class IntervalHeap {
public:
    /**
     * @brief Construct a new IntervalHeap object
     */
    constexpr IntervalHeap() = default;
    /**
     * @brief Construct a new Interval Heap object
     * 
     * @param comp comparator to be used
     * @param cont container with elements 
     */
    constexpr explicit  IntervalHeap(const Compare& comp, const Container & cont = Container()): _comp(comp), _data(cont) {
        heapify();
    }
    /**
     * @brief Construct a new Interval Heap object
     * 
     * @param comp comparator to be used
     * @param cont container with elements 
     */
    constexpr explicit IntervalHeap(const Compare& comp, Container && cont): _comp(comp), _data(std::move(cont)) {
        heapify();
    }
    /**
     * @brief Construct a new Interval Heap object
     * 
     * @param cont container with elements 
     */
    constexpr explicit IntervalHeap(const Container& cont) : IntervalHeap(Compare(), cont) {}
    /**
     * @brief Construct a new Interval Heap object
     * 
     * @param cont container with elements 
     */
    constexpr explicit IntervalHeap(Container&& cont) : IntervalHeap(Compare(), std::move(cont)) {}
    /**
     * @brief Construct a new Interval Heap object
     * 
     * @tparam It iterator to some container with elements T
     * @param first begin iterator
     * @param last end iterator
     * @param comp comparator to be used
     */
    template <class It>
    constexpr IntervalHeap(It first, It last, const Compare& comp = Compare()) : IntervalHeap(comp, Container(first, last)) {}
    /**
     * @brief Return the minimal element in heap, O(1)
     * 
     * @return const reference to the minimal element in heap
     */
    [[nodiscard]] constexpr const T& min() const {
        assert(!empty());
        return _data[ROOT];
    }
    /**
     * @brief Return the minimal element in heap, O(1)
     * 
     * @note WARNING! changing the element might break
     * the heap structure. Use at your own risk
     * 
     * @return const reference to the minimal element in heap
     */
    [[nodiscard]] constexpr T& min() {
        assert(!empty());
        return _data[ROOT];
    }
    /**
     * @brief Return the maximal element in heap, O(1)
     * 
     * @return const reference to the maximal element in heap
     */
    [[nodiscard]] constexpr const T& max() const {
        assert(!empty());
        return size() > 1 ? _data[ROOT + 1] : _data[ROOT];
    }
    /**
     * @brief Return the maximal element in heap, O(1)
     * 
     * @note WARNING! changing the element might break
     * the heap structure. Use at your own risk
     * 
     * @return const reference to the maximal element in heap
     */
    [[nodiscard]] constexpr T& max() {
        assert(!empty());
        return size() > 1 ? _data[ROOT + 1] : _data[ROOT];
    }
    /**
     * @brief Return whether heap is empty or not
     * 
     * @return true if heap is empty
     * @return false if heap is not empty
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return _data.empty();
    }
    /**
     * @brief Return number of elements in heap
     * 
     * @return number of elements in heap
     */
    [[nodiscard]] constexpr size_t size() const noexcept {
        return _data.size();
    }
    /**
     * @brief Insert new element into heap, O(log(n))
     * 
     * @param elem element to be inserted
     */
    constexpr void push(const T& elem) {
        _data.push_back(elem);
        bubble_up(_data.size() - 1);
    }
    /**
     * @brief Insert new element into heap, O(log(n))
     * 
     * @param elem element to be inserted
     */
    constexpr void push(T&& elem) {
        _data.push_back(std::move(elem));
        bubble_up(_data.size() - 1);
    }
    /**
     * @brief Emplace new element into heap, O(log(n))
     * 
     * @param args arguments for constructor of T
     */
    template<class... Args >
    constexpr void emplace(Args&&... args) {
        _data.emplace_back(std::forward<Args>(args)...);
        bubble_up(_data.size() - 1);
    }
    /**
     * @brief Erase minimal element from the heap, O(log(n))
     */
    constexpr void pop_min() {
        assert(!empty());
        size_t n = _data.size();
        size_t idx = ROOT;
        if (n % 2) {
            _data[idx] = std::move(_data[n - 1]);
        } else {
            _data[idx] = std::move(_data[n - 2]);
            _data[n - 2] = std::move(_data[n - 1]);
        }
        _data.pop_back();
        balance_node_check(idx);
        bubble_down_min(idx);
    }
    /**
     * @brief Erase maximal element from the heap, O(log(n))
     */
    constexpr void pop_max() {
        assert(!empty());
        size_t n = _data.size();
        if (n == 1) {
            _data.pop_back();
            return;
        }
        size_t idx = ROOT + 1;
        _data[idx] = std::move(_data[n - 1]);
        _data.pop_back();
        if (n > 2) {
            balance_node(ROOT);
            bubble_down_max(idx);
        }
    }
    /**
     * @brief Replace minimal value with given value, O(log(n))
     * 
     * Offer a faster alternative to calling .pop_min() followed by .push()
     * 
     * @param val value to be inserted
     */
    constexpr void replace_min(const T& val) {
        assert(!empty());
        size_t idx = ROOT;
        _data[idx] = val;
        balance_node_check(idx);
        bubble_down_min(idx);
    }
    /**
     * @brief Replace minimal value with given value, O(log(n))
     * 
     * Offer a faster alternative to calling .pop_min() followed by .push()
     * 
     * @param val value to be inserted
     */
    constexpr void replace_min(T&& val) {
        assert(!empty());
        size_t idx = ROOT;
        _data[idx] = std::move(val);
        balance_node_check(idx);
        bubble_down_min(idx);
    }
    /**
     * @brief Replace maximal value with given value, O(log(n))
     * 
     * Offer a faster alternative to calling .pop_max() followed by .push()
     * 
     * @param val value to be inserted
     */
    constexpr void replace_max(const T& val) {
        assert(!empty());
        if (_data.size() == 1) {
            _data[ROOT] = val;
        } else {
            _data[ROOT + 1] = val;
            balance_node(ROOT);
            bubble_down_max(ROOT + 1);
        }
    }
    /**
     * @brief Replace maximal value with given value, O(log(n))
     * 
     * Offer a faster alternative to calling .pop_max() followed by .push()
     * 
     * @param val value to be inserted
     */
    constexpr void replace_max(T&& val) {
        assert(!empty());
        if (_data.size() == 1) {
            _data[ROOT] = std::move(val);
        } else {
            _data[ROOT + 1] = std::move(val);
            balance_node(ROOT);
            bubble_down_max(ROOT + 1);
        }
    }
    /**
     * @brief Swap content of this with other
     * 
     * @param other IntervalHeap to switch content with
     */
    constexpr void swap(IntervalHeap& other) noexcept(std::is_nothrow_swappable_v<Container> && std::is_nothrow_swappable_v<Compare>) {
        using std::swap;
        swap(_data, other._data);
        swap(_comp, other._comp);
    }
    /**
     * @brief Swap content of two IntervalHeaps
     * 
     * @param lhs first IntervalHeap
     * @param rhs second IntervalHeap
     */
    friend constexpr void swap(IntervalHeap& lhs, IntervalHeap& rhs) noexcept(std::is_nothrow_swappable_v<Container> && std::is_nothrow_swappable_v<Compare>) {
        lhs.swap(rhs);
    }
    /**
     * @brief Reserve capacity for underlying container
     * 
     * @param cap capacity to be reserved
     */
    constexpr void reserve(size_t cap) {
        _data.reserve(cap);
    }
    /**
     * @brief Clears the heap
     */
    constexpr void clear() {
        _data.clear();
    }
private:
    static constexpr const size_t ROOT = 0;
    [[no_unique_address]] Compare _comp;
    Container _data;

    static constexpr size_t get_parent(size_t idx) noexcept {
        return (idx - 2) / 4 * 2;
    }
    static constexpr size_t get_left(size_t idx) noexcept {
        return (idx + 1) * 2;
    }
    static constexpr bool is_min(size_t idx) noexcept {
        return idx % 2 == 0;
    }
    static constexpr bool is_max(size_t idx) noexcept {
        return idx % 2 == 1;
    }

    /**
     * @brief Standard bubble up, O(log(n))
     * 
     * @param idx index of element to bubble up
     */
    constexpr void bubble_up(size_t idx) {
        assert(_data.size() > idx);
        assert(idx >= ROOT);
        T cur = std::move(_data[idx]);
        
        // Fix the interval in curent node
        if (is_max(idx) && _comp(cur, _data[idx - 1])) {
            _data[idx] = std::move(_data[idx - 1]);
            idx--;
        }
        size_t par = get_parent(idx);
        // cur is lower than the parent min - insert into the min heap
        if (idx > ROOT + 1 && _comp(cur, _data[par])) {
            do {
                _data[idx] = std::move(_data[par]);
                idx = par;
                par = get_parent(idx);
            } while (idx > ROOT + 1 && _comp(cur, _data[par]));
        // cur is higher than the parent max - insert into the max heap
        } else if (idx > ROOT + 1 && _comp(_data[par + 1], cur)) {
            // par must be odd so we look at max value
            par++;
            do {
                _data[idx] = std::move(_data[par]);
                idx = par;
                par = get_parent(idx) + 1;
            } while (idx > ROOT + 1 && _comp(_data[par], cur));
        }
        _data[idx] = std::move(cur);
    }
    /**
     * @brief Standard bubble down bubbling min indexes, O(log(n))
     * 
     * Bubble down assume that the curent node (interval) is ordered
     * as per the rules, meaning _data[idx] < _data[idx + 1] has to be TRUE
     * 
     * @param idx index of element to bubble down
     */
    constexpr void bubble_down_min(size_t idx) {
        assert(_data.size() > idx || idx == 0);
        assert(is_min(idx));
        assert(idx >= ROOT);
        using std::swap;
        size_t child = get_left(idx);
        size_t n = _data.size();
        while (child < n) {
            // choose the smaller child, consider only min values
            // +2 to acces right child
            if (child + 2 < n && _comp(_data[child + 2], _data[child]))
                child += 2;
            // if child is smaller, swap and continue
            if (_comp(_data[child], _data[idx])) {
                swap(_data[idx], _data[child]);
                // if node interval property is not satisfied, swap them
                if (child + 1 < n && _comp(_data[child + 1], _data[child]))
                    swap(_data[child + 1], _data[child]);
                idx = child;
                child = get_left(idx);
            } else {
                break;
            }
        }
    }
    /**
     * @brief Standard bubble down bubbling max indexes, O(log(n))
     * 
     * Bubble down assume that the curent node (interval) is ordered
     * as per the rules, meaning _data[idx - 1] < _data[idx] has to be TRUE
     * 
     * @param idx index of element to bubble down
     */
    constexpr void bubble_down_max(size_t idx) {
        assert(is_max(idx));
        assert(idx >= ROOT);
        assert(_data.size() > idx || idx == 0);
        using std::swap;
        idx--;
        size_t child = get_left(idx);
        size_t n = _data.size();
        while (child < n) {
            // need to consider edge case when node consist only of min value
            // this value can still be put as max value in curent node
            size_t child1 = child + 1 < n ? child + 1 : child;
            size_t child2 = child + 3 < n ? child + 3 : child + 2;
            // choose the bigger child, consider only max values
            if (child2 < n && _comp(_data[child1], _data[child2])) {
                child += 2;
                child1 = child2;
            }
            // if the child is bigger, swap them
            // keep in mind that children denotes node the child is in,
            // while child1 denotes the actuall position (min or max)
            if (_comp(_data[idx + 1], _data[child1])) {
                swap(_data[idx + 1], _data[child1]);
                // if node interval property is not satisfied, swap them
                // if max child was in max spot (not min) and is smaller than its min brother...
                if (is_max(child1) && _comp(_data[child1], _data[child1 - 1]))
                    swap(_data[child1], _data[child1 - 1]);
                idx = child;
                child = get_left(idx);
            } else {
                break;
            }
        }
        // check interval property again
        // need to also check the the right side of interval exists
        if (idx + 1 < n && _comp(_data[idx + 1], _data[idx]))
            swap(_data[idx], _data[idx + 1]);
    }
    /**
     * @brief Creates valid heap structure from _data, O(n)
     */
    constexpr void heapify() {
        using std::swap;
        if (_data.size() <= 2) {
            if (_data.size() == 2 && _comp(_data[1], _data[0]))
                swap(_data[1], _data[0]);
            return;
        }
        for (size_t i = 0; i < _data.size() - 1; i += 2) {
            balance_node(i);
        }
        // bubble down all internal nodes
        for (long long i = (_data.size() - 1) / 4 * 2; i >= 0; i -= 2) {
            bubble_down_max(i + 1);
            bubble_down_min(i);
        }
    }
    constexpr void balance_node(size_t idx) {
        using std::swap;
        if (_comp(_data[idx + 1], _data[idx]))
            swap(_data[idx + 1], _data[idx]);
    }
    constexpr void balance_node_check(size_t idx) {
        if (idx + 1 < _data.size())
            balance_node(idx);
    }
};

}; // namespace murty