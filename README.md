# UAV Smoke Deployment Optimization

该项目研究无人机烟幕投放过程中的几何建模与策略优化问题，核心任务是通过参数搜索与部署设计，提高遮蔽效果并延长有效遮蔽时间。

## Project Goals

- 建立投放场景的几何关系模型
- 优化无人机投放位置与策略参数
- 分析不同问题阶段下的收敛行为与方案效果
- 生成题目附件与结果文件，用于论文写作和结果展示

## Methods

- 几何建模
- 遮蔽时间计算
- 差分进化与局部优化
- 并行计算
- 热力图与收敛曲线分析
- Python 与 MATLAB 混合求解

## Repository Structure

- `代码/`
  Python 主求解脚本与结果目录
- `原数据/`
  原始输入数据
- `附件/`
  结果附件
- `matlab/problem_2/`
  第二问 MATLAB 脚本
- `matlab/problem_4/`
  第四问 MATLAB 脚本与说明

## Key Scripts

- `代码/第一问.py`
  第一问几何求解
- `代码/第二问.py`
  第二问优化分析
- `代码/第三问.py`
  第三问主优化程序
- `代码/第三问几何.py`
  第三问几何版本求解
- `代码/第四问.py`
  第四问部署优化
- `代码/第五问.py`
  第五问扩展分析
- `代码/第四问附件生成.py`
  结果附件生成
- `代码/第五问附件生成.py`
  结果附件生成

## Data And Outputs

项目保留了原始数据、生成结果、热力图、收敛曲线、论文图与附件表格，适合直接用于复现实验过程和展示最终方案。

## Running

Python 部分可直接按问题脚本逐个运行；MATLAB 部分可单独执行 `matlab/` 目录下的 `.m` 文件。

## Main Dependencies

- Python：`numpy`、`pandas`、`matplotlib`、`scipy`、`seaborn`、`joblib`
- MATLAB：运行 `.m` 脚本所需环境
