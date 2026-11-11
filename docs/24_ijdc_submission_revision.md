# IJDC 定向修订与投稿状态

核对日期：2026-11-04  
目标期刊：*International Journal of Dynamics and Control* (IJDC)  
官方范围：https://link.springer.com/journal/40435/aims-and-scope  
官方投稿指南：https://link.springer.com/journal/40435/submission-guidelines

## 已完成

- 使用 Springer Nature `sn-jnl` 双栏模板生成匿名正文；PDF 元数据不含作者身份。
- 将论文标题、摘要、关键词、引言、相关工作、讨论与结论重构为非线性动力学、状态估计和预测控制导向。
- 补充并核验 Kalman 滤波、非线性互补滤波、学习型 MPC 和 IJDC 状态观测器文献；另由 arXiv 原始页面复核三条 2025/2026 预印本，修正 HaM-World 作者顺序，并将机器生成的 `arXiv (Cornell University)` 统一规范为 `arXiv preprint`。核验记录见 `literature/ijdc_citation_audit.json`。
- 保留所有实证边界：H1 获支持，H2 未获支持；外部无人机评估中，CFR-WM 的全状态 NRMSE 在全部八个比较中高于未滤波残差模型，而 50 步位置误差在四种上下文中均较低。
- 删除失败候选日志、修订过程叙事、重复免责声明和道歉式措辞，不删除负面结果、统计不确定性或适用范围。
- 数据和代码可用性声明移至独立 Title Page；因尚无公共存档标识符，声明中未填写 DOI。
- 按 Springer 对生成式 AI 的披露要求，在 Methods 说明工具用途；独立 Title Page 同时保留作者责任声明。作者已于 2026-11-04 确认核验引用、公式、代码输出、统计结果和科学表述并承担责任。
- 图件提供 PDF/EPS 矢量版本和 600 dpi PNG 预览；最小字号 8 pt，最小线宽 0.5 pt。图件来源及哈希见 `paper/ijdc/figures/artwork_manifest.json`。

## 验证结果

- 匿名正文：20 页；摘要 236 词；6 个关键词。
- LaTeX/BibTeX 编译：通过；无 overfull box、未定义引用或重复标签。
- 自动化测试：128 项通过；仅有一个既有 PyTorch Transformer 性能提示。
- 视觉检查：逐页渲染检查 20 页正文与 2 页 Title Page，未见空白页、截断、遮挡或不可读图表。
- 匿名源码包：`paper/ijdc/IJDC_anonymous_source.zip`；内容仅含正文、参考文献、模板、四幅 PDF 图、BBL 和编译后 PDF，不含 Title Page。
- 完整性记录：`paper/ijdc/source_manifest.json`、`build_manifest.json`、`package_manifest.json`。

## 作者控制的剩余事项

当前匿名稿和源码包已经可供投稿前审阅，但整个投稿仍不可直接提交。作者需完成：姓名与顺序、单位、通讯作者邮箱、ORCID、Funding、Competing interests、Acknowledgements、CRediT 作者贡献，以及 Ethics/Consent 的最终确认。清单与建议措辞见 `paper/ijdc/author_information_required.md`；独立 Title Page 草稿见 `paper/ijdc/title_page.tex`。

如投稿时提供匿名复现材料，需先建立符合数据许可和双盲要求的匿名访问位置；取得真实 URL 或 DOI 后再更新声明。不得以本地材料、哈希清单或编译通过代替公共存档状态。
