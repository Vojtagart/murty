/**
 * @file      binary_heap.hpp
 * @brief     Implements binary heap
 * @author    @vojtagart
 * @date      21/02/2026
 * @see       https://github.com/Vojtagart/murty
 * * @par Credits & Acknowledgments
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
 * @brief Minimal binary heap
 * 
 * Efficient implementation of Minimal binary heap
 * 
 * @tparam T - the type of the stored elements
 * @tparam Container - the type of underlying container to store elements
 * @tparam Compare - a class providing a strict weak ordering
 */
template <typename T, class Container=std::vector<T>, class Compare=std::less<typename Container::value_type>>
class BinaryHeap {
public:
    /**
     * @brief Construct a new BinaryHeap object
     */
    constexpr BinaryHeap() = default;
    /**
     * @brief Construct a new Binary Heap object
     * 
     * @param comp comparator to be used
     * @param cont container with elements
     */
    constexpr explicit BinaryHeap(const Compare& comp, const Container& cont = Container()): _comp(comp), _data(cont) {
        heapify();
    }
    /**
     * @brief Construct a new Binary Heap object
     * 
     * @param comp comparator to be used
     * @param cont container with elements
     */
    constexpr explicit BinaryHeap(const Compare& comp, Container&& cont): _comp(comp), _data(std::move(cont)) {
        heapify();
    }
    /**
     * @brief Construct a new Binary Heap object
     * 
     * @param cont container with elements
     */
    constexpr explicit BinaryHeap(const Container& cont) : BinaryHeap(Compare(), cont) {}
    /**
     * @brief Construct a new Binary Heap object
     * 
     * @param cont container with elements
     */
    constexpr explicit BinaryHeap(Container&& cont) : BinaryHeap(Compare(), std::move(cont)) {}
    /**
     * @brief Construct a new Binary Heap object
     * 
     * @tparam It iterator to some container with elements T
     * @param first begin iterator
     * @param last end iterator
     * @param comp comparator to be used
     */
    template <class It>
    constexpr BinaryHeap(It first, It last, const Compare& comp = Compare()) : BinaryHeap(comp, Container(first, last)) {}
    /**
     * @brief Return the minimal element in heap, O(1)
     * 
     * @return const reference to the minimal element in heap
     */
    [[nodiscard]] constexpr const T& top() const {
        assert(!empty());
        return _data[ROOT];
    }
    /**
     * @brief Return the minimal element in heap, O(1)
     * 
     * @note WARNING! changing the element might break
     * the heap structure. Use at your own risk
     * 
     * @return reference to the minimal element in heap
     */
    [[nodiscard]] constexpr T& top() {
        assert(!empty());
        return _data[ROOT];
    }
    /**
     * @brief Return the minimal element in heap, O(1)
     * 
     * @return reference to the minimal element in heap
     */
    [[nodiscard]] constexpr const T& min() const {
        return top();
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
        return top();
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
     * @brief Return minimal element from the heap, O(log(n))
     * 
     * Works by replacing the top element with it's smaller child
     * until we get to the leaf, moving a hole after the minimal element
     * there this way. Then we swap it with the right-most leaf and bubble it up.
     * This makes the average number of comparisons needed smaller as than the standard
     * swap with the right-most leaf and bubble down as the leaf has a high chance of
     * bubbling all the way down, thus needing 2 * log2(n) comparisons.
     */
    constexpr void pop() {
        assert(!empty());

        // Older version
        // using std::swap;
        // swap(_data[ROOT], _data.back());
        // _data.pop_back();
        // bubble_down(ROOT);
    
        size_t idx = move_hole_down(ROOT);
        if (idx + 1 == _data.size()) {
            _data.pop_back();
        } else {
            _data[idx] = std::move(_data.back());
            _data.pop_back();
            bubble_up(idx);
        }
    }
    /**
     * @brief Replace minimal value with given value, O(log(n))
     * 
     * Offer a faster alternative to calling .pop() followed by .push()
     * 
     * @param val value to be inserted
     */
    constexpr void replace_top(const T & val) {
        assert(!empty());
        _data[ROOT] = val;
        bubble_down(ROOT);
    }
    /**
     * @brief Replace minimal value with given value, O(log(n))
     * 
     * Offer a faster alternative to calling .pop() followed by .push()
     * 
     * @param val value to be inserted
     */
    constexpr void replace_top(T && val) {
        assert(!empty());
        _data[ROOT] = std::move(val);
        bubble_down(ROOT);
    }
    /**
     * @brief Alias for replace_top, O(log(n))
     * 
     * @param val value to be inserted
     */
    constexpr void replace_min(const T & val) {
        replace_top(val);
    }
    /**
     * @brief Alias for replace_top, O(log(n))
     * 
     * @param val value to be inserted
     */
    constexpr void replace_min(T && val) {
        replace_top(std::move(val));
    }
    /**
     * @brief Swap content of this with other
     * 
     * @param other BinaryHeap to switch content with
     */
    constexpr void swap(BinaryHeap& other) noexcept(std::is_nothrow_swappable_v<Container> && std::is_nothrow_swappable_v<Compare>) {
        using std::swap;
        swap(_data, other._data);
        swap(_comp, other._comp);
    }
    /**
     * @brief Swap content of two BinaryHeaps
     * 
     * @param lhs first BinaryHeap
     * @param rhs second BinaryHeap
     */
    friend constexpr void swap(BinaryHeap& lhs, BinaryHeap& rhs) noexcept(std::is_nothrow_swappable_v<Container> && std::is_nothrow_swappable_v<Compare>) {
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
        return (idx - 1) / 2;
    }
    static constexpr size_t get_left(size_t idx) noexcept {
        return 2 * idx + 1;
    }

    /**
     * @brief Standard bubble up, O(log(n))
     * 
     * @param idx index of element to bubble up
     */
    constexpr void bubble_up(size_t idx) {
        assert(idx >= ROOT);
        assert(idx < _data.size());
        T cur = std::move(_data[idx]);
        while (idx > ROOT) {
            size_t par = get_parent(idx);
            if (!_comp(cur, _data[par])) break;
            _data[idx] = std::move(_data[par]);
            idx = par;
        }
        _data[idx] = std::move(cur);
    }
    /**
     * @brief Standard bubble down, O(log(n))
     * 
     * @param idx index of element to bubble down
     */
    constexpr void bubble_down(size_t idx) {
        assert(idx >= ROOT);
        assert(idx < _data.size());
        size_t n = _data.size();
        T cur = std::move(_data[idx]);
        size_t child = get_left(idx);
        while (child < n) {
            if (child + 1 < n && _comp(_data[child + 1], _data[child]))
                child++;
            if (_comp(_data[child], cur)) {
                _data[idx] = std::move(_data[child]);
                idx = child;
            } else {
                break;
            }
            child = get_left(idx);
        }
        _data[idx] = std::move(cur);
    }
    /**
     * @brief moves hole (place with missing element) in the tree downwards, O(log(n))
     * 
     * Works by replacing the hole with smaller child and moving
     * the hole into the leaf. After it reaches leaf, iot returns its index
     * 
     * @param idx curent index of the hole
     * @return index where the hole was moved
     */
    constexpr size_t move_hole_down(size_t idx) {
        assert(idx >= ROOT);
        assert(idx < _data.size());
        size_t child = get_left(idx);
        size_t n = _data.size();
        while (child < n) {
            if (child + 1 < n && _comp(_data[child + 1], _data[child]))
                child++;
            _data[idx] = std::move(_data[child]);
            idx = child;
            child = get_left(idx);
        }
        return idx;
    }
    /**
     * @brief Creates valid heap structure from _data, O(n)
     */
    constexpr void heapify() {
        for (long long i = static_cast<long long>(_data.size()) / 2 - 1; i >= 0; i--) {
            bubble_down(i);
        }
    }
};

}; // namespace murty