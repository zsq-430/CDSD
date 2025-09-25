# Generalized Document Tampering Localization via Color and Semantic Disentanglement  
# 通过颜色与语义解耦的广义文档篡改定位  

**Code is already public**  
**代码已公开**  

## 项目结构 / Project Structure
```
DTL_CDSD/
├── code/                           # 主要代码目录
│   ├── dtd.py                      # 核心模型定义文件
│   ├── dtdtrain_CD.py              # CD训练脚本
│   ├── dtdEval.py                  # CD评估脚本
│   ├── dtdEval_npy_ours.py         # 用于输出npy，供SD模块使用
│   ├── data_loader.py              # 数据加载器
│   ├── fph.py                      # 频率感知头(Frequency Perception Head)
│   ├── swins.py                    # Swin Transformer实现
│   ├── losses/                     # 损失函数模块
│   │   ├── __init__.py
│   │   ├── lovasz.py               # Lovász损失
│   │   ├── soft_ce.py              # 软交叉熵损失
│   │   └── ...                     # 其他损失函数
│   ├── SD_Semantic_Disentanglement/ # 语义解离模块
│   │   ├── 3_npy_clustering.py     # SD聚类工具
│   │   └── mdb_json/               # 元数据JSON文件
│   ├── checkpoint/                 # 模型检查点
│   │   └── model_load/             # 预训练模型
│   ├── pks/                        # 量化表数据
│   └── tool/                       # 工具文件
│       └── DTD_MedianColor.json   # 中值颜色配置
└── qt_table.pk                     # 量化表文件
```

---

## 📖 Overview | 项目概述  
This repository contains the official implementation of the paper:  
**Generalized Document Tampering Localization via Color and Semantic Disentanglement**  
本仓库为论文 **[通过颜色与语义解耦的广义文档篡改定位](论文链接)** 的官方实现代码。

> 🔥 **Key Idea**: We propose a generalized framework that disentangles **color patterns** and **semantic context** to localize multi-type document tampering (text alteration, image forgery, layout manipulation, etc.).  
> 🔥 **核心创新**: 通过解耦文档图像的**颜色特征**与**语义上下文**，实现对文本篡改、图像伪造、版面篡改等多类型篡改的广义定位。

---

## 🚀 Features | 核心特点  
- **Color Disentanglement**: Eliminates color interference for robust texture analysis  
- **Semantic Consistency**: Preserves document structure under complex manipulations  
- **Cross-domain Generalization**: Adapts to scanned/printed/digital-born documents  

- **颜色解耦**: 消除色彩干扰，增强纹理特征鲁棒性  
- **语义一致性**: 保持复杂篡改下的文档结构特征  
- **跨域泛化**: 兼容扫描/印刷/数字原生文档  

---

## 🛠️ Technical Overview | 技术框架  
![model](image/image.png)

---

## ⏳ Coming Soon | 即将发布  
We are optimizing code readability and preparing evaluation protocols.  
我们正在优化代码可读性并准备标准化评估协议。  

**Expected Release**: Q4 2025  
**预计时间**: 2025年第四季度  

---

## 📚 Citation | 引用  
```bibtex
@article{yourpaper2025,
  title={Generalized Document Tampering Localization via Color and Semantic Disentanglement},
  author={Your Name, Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
