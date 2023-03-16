# 降水订正实验
20230110 <br>
降水订正实验，尝试使用一维数据进行建模。

## 程序结构
src:源码 <br>
z01checkdata.py 验证数据提取结果都正确性。 <br>
p01organizedata.py 组织实验数据。 <br>
p02model.py 模型定义。 <br>
p03trainandtest.py 模型训练和测试实验(lstm)。 <br>
p04trainandtest.py 模型训练和测试实验(ann)。 <br>
p05organizeresultdata.py 将结果数据组织成nc文件。 <br>
p07model.py 使用ANN+LSTM混合建模，一维数据。 <br>
p08trainandtest.py 对应py07模型的训练和测试程序。 <br>


data:数据 <br>
result:结果 <br>

## log
20230116 组织数据：单个case对一次可以生成未来数个月的数据，每个格点提取数据，形成一维序列，作为输入。进行建模。 <br>
20230120 基础LSTM实验。 <br>
20230123 基础ANN实验。 <br>
20230124 结果数据组织为nc文件。 <br>


