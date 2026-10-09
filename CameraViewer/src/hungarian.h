#ifndef HUNGARIAN_H
#define HUNGARIAN_H

/**
 * @file hungarian.h
 * @brief 匈牙利算法（Kuhn-Munkres）——求解「全局最优指派」问题
 *
 * 用途：当同一时刻有多个下车探测（probe）需要与多个在车乘客模板（template）
 * 逐一配对时，逐个取最大（贪心）会因“先到先得”而互相抢占，产生次优解。
 * 匈牙利算法在 O(n^3) 内给出**全局代价最小**（即整体相似度最大）的一一指派，
 * 显著降低“多人同时上下车”场景下的错配率。
 *
 * 说明：内部要求 行数 <= 列数；当探测数多于模板数时自动转置求解。
 */

#include <vector>
#include <limits>
#include <algorithm>

/**
 * @brief 最小代价指派（行数 n <= 列数 m）
 * @param cost n×m 代价矩阵
 * @return 长度 n 的数组，assign[i] = 第 i 行匹配到的列（0-based）
 */
inline std::vector<int> hungarianSolve(const std::vector<std::vector<double>> &cost)
{
    const int n = static_cast<int>(cost.size());
    if (n == 0) return {};
    const int m = (n > 0) ? static_cast<int>(cost[0].size()) : 0;
    if (m == 0) return std::vector<int>(n, -1);

    const double INF = std::numeric_limits<double>::max() / 4.0;

    // u/v: 行/列势能；p: 列匹配的行；way: 增广路径回溯
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, INF);
        std::vector<char> used(m + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            int j1 = -1;
            double delta = INF;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<int> assign(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] > 0) assign[p[j] - 1] = j - 1;
    }
    return assign;
}

/**
 * @brief 最大相似度的全局指派（行 = 探测，列 = 模板）
 * @param sim n×m 相似度矩阵（越大越相似）
 * @return 长度 n 的数组，assign[i] = 第 i 个探测匹配到的模板列（-1 表示未匹配）
 */
inline std::vector<int> hungarianAssignMax(const std::vector<std::vector<double>> &sim)
{
    const int n = static_cast<int>(sim.size());
    if (n == 0) return {};
    const int m = (n > 0) ? static_cast<int>(sim[0].size()) : 0;
    if (m == 0) return std::vector<int>(n, -1);

    if (n <= m) {
        std::vector<std::vector<double>> cost(n, std::vector<double>(m, 1.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) cost[i][j] = 1.0 - sim[i][j];
        return hungarianSolve(cost);
    }

    // 探测数 > 模板数：转置后求解，再映射回「探测 -> 模板」
    std::vector<std::vector<double>> cost(m, std::vector<double>(n, 1.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) cost[j][i] = 1.0 - sim[i][j];
    const std::vector<int> colAssign = hungarianSolve(cost);   // 模板 -> 探测
    std::vector<int> assign(n, -1);
    for (int j = 0; j < m; ++j) {
        const int i = colAssign[j];
        if (i >= 0 && i < n) assign[i] = j;
    }
    return assign;
}

#endif // HUNGARIAN_H
