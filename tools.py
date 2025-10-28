def longest_common_substring(s1, s2):
    """返回 s1 和 s2 之间的最长公共子串及其长度"""
    m, n = len(s1), len(s2)
    max_len = 0
    ending_index_s1 = 0
    
    # 创建一个二维数组保存子串长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 填充 dp 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    ending_index_s1 = i
    
    # 提取最长公共子串
    longest_substring = s1[ending_index_s1 - max_len : ending_index_s1]
    
    return longest_substring, max_len

def iterative_lcs_similarity(s1, s2):
    """迭代计算字符串相似度"""
    # 确保 s1 是较短的字符串
    if len(s1) == 0 or len(s2) == 0:
        return 0
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    min_len = len(s1)
    total_lcs_len = 0
    while s1:
        # 找到最长公共子串
        lcs, lcs_len = longest_common_substring(s1, s2)
        
        if lcs_len == 0:
            break
        
        # 计算比值
        total_lcs_len += lcs_len
        
        # 从 s1 中移除已匹配的子串
        s1 = s1.replace(lcs, '', 1)
    
    return total_lcs_len / min_len
