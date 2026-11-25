# 雨夹雪 · 译文漫游 (`tm-feed`)

基于 [L-M-Sherlock/tm-index](https://github.com/L-M-Sherlock/tm-index) 与 [L-M-Sherlock/ZhiHuArchive](https://github.com/L-M-Sherlock/ZhiHuArchive) 构建的**类知乎双体裁个性化推荐阅读器**，视觉与排版风格复刻自个人博客 [mizore.blog](https://mizore.blog)。

- 在线访问：<https://mizorewww.github.io/tm-feed/>

## 核心特性

1. **全量离线正文收录（3,580 篇）**：
   - 收录 **2,187 篇精校译文**（总计 1,024 万字）与 **1,393 篇关联知乎高赞回答**于 `public/posts/<id>.json`，点击卡片站内秒开全文，免跳出登录墙。
   - 正文内互引链接自动识别，鼠标悬停即刻弹出**站内译文摘要预览浮窗**，原生知乎参考注释自动转换为双向脚注。
2. **针对 91.99% 同作者孤岛效应的混合推荐引擎**：
   - **离线特征（`scripts/build_corpus.py`）**：融合字符/词级 N-gram TF-IDF 与「译文引用 + 知乎回答共被引（Adamic-Adar）」图谱，生成 `128` 维 `Int8` 量化混合嵌入（`public/data/embeddings.bin`，仅约 450 KB）。
   - **实时双尺度画像**：纯本地 `localStorage` 记录点击、有效阅读停留时长、滚动完读率、感兴趣、稍后再读与负反馈屏蔽，实时更新短期意图向量与长期偏好向量。
   - **MMR 滑动窗口多样性重排**：打散同作者与同专栏连续卡片，按合理节奏混排深度长译文与短平快知乎问答。
3. **文末递进式三栏延伸 + 无限续刷流**：
   - 每篇文章底部同时提供 **「顺藤摸瓜（上下文与直接互引）」**、**「跨界共鸣（强制异作者语义/共引破茧）」**、**「知乎实战（引用本文的高赞回答）」**，继续下滑即可无缝接着刷。

## 本地预览与更新语料

```sh
# 本地预览静态站点
python3 -m http.server 8080 -d public

# 从上游 tm-index 与 ZhiHuArchive 重新构建语料与推荐向量
uv run --with numpy --with scipy --with scikit-learn --with beautifulsoup4 --with lxml python3 scripts/build_corpus.py
```

## 协议与署名

详见 [LICENSE](LICENSE) 与 [NOTICE.md](NOTICE.md)。索引组织与站点代码按 **CC0 1.0 Universal** 发布；各篇译文所对应的原著与译文权利仍归各原作者及译者所有。
