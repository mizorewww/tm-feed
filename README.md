# 译文漫游 (`tm-feed`)

基于 [L-M-Sherlock/tm-index](https://github.com/L-M-Sherlock/tm-index) 与 [L-M-Sherlock/ZhiHuArchive](https://github.com/L-M-Sherlock/ZhiHuArchive) 构建的极简推荐阅读器。

- 在线访问：<https://mizorewww.github.io/tm-feed/>

## 设计原则

1. **极简无打扰**：无导航栏、无看板、无点赞/不感兴趣等冗余按钮；顶部仅保留一个即时搜索框，**点进去阅读即代表感兴趣**（自动结合停留时长与滚动深度记录权重）。
2. **双场景明确分工**：
   - **主页推荐流**：纯根据你的历史阅读轨迹推荐（融合 128 维语义向量、图谱一度扩散与 MMR 同作者打散）。
   - **文末延伸阅读**：采用与主页一致的极简卡片样式，固定推荐 **6 篇**，同时权衡**当前文章内容相关性（语义 + 引用/共引关系）**与**个人历史阅读偏好**。
3. **全量同源存档**：3,580 篇译文与知乎回答全文均存于 `public/posts/<id>.json`，点击秒开。

## 协议与署名

详见 [LICENSE](LICENSE) 与 [NOTICE.md](NOTICE.md)。索引与站点代码遵循 **CC0 1.0 Universal**；各篇译文及原著权利归原作者与译者所有。
