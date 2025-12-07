library(dplyr)
library(ggplot2)
library(ggpubr)
library(readr)
library(tidyverse)
library(pheatmap)




setwd("E:/sgrna_merge_cas_naa_ngg")
data1_1<-read.csv("tem_1_filter.csv",header=TRUE)
df_filtered_1_1 <- data1_1%>%
  filter(!is.na(ispymac) & !is.na(spycas9) & !is.na(spymac))


data1_2<-read.csv("tem_2_filter.csv",header=TRUE)
df_filtered_1_2 <- data1_2%>%
  filter(!is.na(ispymac) & !is.na(spycas9) & !is.na(spymac))


data1_3<-read.csv("tem_3_filter.csv",header=TRUE)
df_filtered_1_3 <- data1_3%>%
  filter(!is.na(ispymac) & !is.na(spycas9) & !is.na(spymac))




data1_4<-read.csv("tem_4_filter.csv",header=TRUE)
df_filtered_1_4 <- data1_4%>%
  filter(!is.na(ispymac) & !is.na(spycas9) & !is.na(spymac))




names(df_filtered_1_1)[-1] <- paste0(names(df_filtered_1_1)[-1], "_1bp")
names(df_filtered_1_2)[-1] <- paste0(names(df_filtered_1_2)[-1], "_2bp")
names(df_filtered_1_3)[-1] <- paste0(names(df_filtered_1_3)[-1], "_3bp")
names(df_filtered_1_4)[-1] <- paste0(names(df_filtered_1_4)[-1], "_4bp")


data_all <- reduce(list(df_filtered_1_1, df_filtered_1_2, df_filtered_1_3, df_filtered_1_4), full_join, by = "sgrna")


data_long <- data_all %>%
  pivot_longer(
    cols = -sgrna,
    names_to = c("group", "bp"),
    names_sep = "_",
    values_to = "value"
  )


data_wide <- data_long %>%
  pivot_wider(
    names_from = bp,
    values_from = value
  ) %>%
  mutate(`5bp` = 1 - (`1bp` + `2bp` + `3bp` + `4bp`))


group_list <- split(data_wide, data_wide$group)




# 指定列顺序
col_order <- c("5bp", "1bp", "2bp", "3bp", "4bp")


# 聚类簇数
k_clusters <- 5


for (grp in names(group_list)) {
  cat("\n==== 处理", grp, "====\n")
  
  df <- group_list[[grp]]
  
  # --- Step 1. 构建矩阵 ---
  mat_raw <- df %>%
    select(all_of(col_order)) %>%
    as.matrix()
  rownames(mat_raw) <- df$sgrna
  
  # --- Step 2. 确定 dominant_bp ---
  dominant_bp <- apply(mat_raw, 1, function(x) {
    if (all(is.na(x))) return(NA_character_)
    names(x)[which.max(x)]
  })
  df$dominant_bp <- dominant_bp
  
  # --- Step 3. 按 dominant_bp 排序 ---
  # 定义排序顺序：先5bp，其次1bp-4bp
  df <- df %>%
    mutate(dominant_bp = factor(dominant_bp, levels = col_order)) %>%
    arrange(dominant_bp)
  
  mat_ordered <- df %>%
    select(all_of(col_order)) %>%
    as.matrix()
  rownames(mat_ordered) <- df$sgrna
  
  # --- Step 4. 在每个 dominant_bp 内做模式聚类 ---
  # 我们将每个 dominant_bp 分组后分别计算聚类顺序
  ordered_rows <- c()
  for (bp in col_order) {
    sub_mat <- mat_ordered[df$dominant_bp == bp, , drop = FALSE]
    if (nrow(sub_mat) > 1) {
      # 计算相关性距离并聚类
      hc <- hclust(as.dist(1 - cor(t(sub_mat), use = "pairwise.complete.obs")), method = "average")
      ordered_rows <- c(ordered_rows, rownames(sub_mat)[hc$order])
    } else {
      ordered_rows <- c(ordered_rows, rownames(sub_mat))
    }
  }
  
  mat_final <- mat_ordered[ordered_rows, ]
  df <- df[match(rownames(mat_final), df$sgrna), ]
  
  # --- Step 5. 聚类结果提取与统计 ---
  hc_all <- hclust(as.dist(1 - cor(t(mat_final), use = "pairwise.complete.obs")), method = "average")
  cluster_assign <- cutree(hc_all, k = k_clusters)
  df$cluster <- cluster_assign
  
  cluster_summary <- df %>%
    count(cluster, dominant_bp) %>%
    group_by(cluster) %>%
    mutate(prop = n / sum(n)) %>%
    ungroup()
  
  print(cluster_summary)
  
  # --- Step 6. 绘图（固定列顺序 + 行按 dominant_bp 聚类） ---
  pdf(paste0("heatmap_pattern_ordered_", grp, ".pdf"), width = 6, height = 8)
  pheatmap(
    mat_final,
    cluster_rows = FALSE,     # 禁止重新聚类行（保持我们指定顺序）
    cluster_cols = FALSE,     # 列顺序固定
    scale = "row",
    main = paste("Pattern-based clustering (ordered) -", grp)
  )
  dev.off()
  
  # --- Step 7. 保存结果 ---
  write.csv(df, paste0("cluster_result_pattern_ordered_", grp, ".csv"), row.names = FALSE)
  write.csv(cluster_summary, paste0("cluster_summary_pattern_ordered_", grp, ".csv"), row.names = FALSE)
}
