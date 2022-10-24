import matplotlib.pyplot as plt
import torch
from NN_CONST import *
from utils import OtherUtils, PaintUtils
from torch.utils.data import Dataset, DataLoader
import utils
from NN import NN


class TestDataset(Dataset):
    def __init__(self, TEST_START_YEAR, TEST_END_YEAR):
        super().__init__()
        self.case_data = torch.Tensor(
            utils.CaseParser.get_many_2d_pravg(CASE_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA))
        self.obs_data = torch.Tensor(
            utils.ObsParser.get_many_2d_pravg(OBS_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA, MONTHS))

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


def test():
    corr_cases, test_cases, test_obses = [], [], []
    for TEST_YEAR in range(1991, 2020):
        # 读取模型
        model = NN()
        model.load_state_dict(torch.load(f"./models/1991-2019年模型(除{TEST_YEAR}年).pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载测试集
        test_dataset = TestDataset(TEST_YEAR, TEST_YEAR)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                print(f"{TEST_YEAR}年")
                print("订正前mse", utils.OtherUtils.mse(inputs, labels))
                print("订正后mse", utils.OtherUtils.mse(outputs, labels))

                for i in range(0, 1):
                    test_cases.append(inputs[0][i].cpu().numpy())
                    corr_cases.append(outputs[0][i].cpu().numpy())
                    test_obses.append(labels[0][i].cpu().numpy())

                    plt.rcParams['font.family'] = ['SimHei']
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax3 = fig.add_subplot(1, 3, 3)

                    ax1.set_title("订正前")
                    ax1.imshow(inputs[0][i])

                    ax2.set_title("订正后")
                    ax2.imshow(outputs[0][i])

                    ax3.set_title("obs")
                    ax3.imshow(labels[0][i])
                    plt.savefig(f"./results/{TEST_YEAR}年{i+4}月")
                    # plt.show()
                    plt.close()

    # TCC相关
    corr_tcc = OtherUtils.cal_TCC(corr_cases, test_obses)
    case_tcc = OtherUtils.cal_TCC(test_cases, test_obses)
    tcc_img = PaintUtils.paint_TCC(case_tcc, corr_tcc)
    tcc_img.savefig(f"./评价指标/TCC/91-19年4月 test")
    tcc_img.close()

    corr_acc = OtherUtils.cal_ACC(corr_cases, test_obses)
    case_acc = OtherUtils.cal_ACC(test_cases, test_obses)
    acc_img = PaintUtils.paint_ACC(range(1991, 2020), case_acc, corr_acc)
    acc_img.savefig(f"./评价指标/ACC/91-19年4月 test")
    acc_img.close()



if __name__ == '__main__':
    plt.rcParams['axes.unicode_minus'] = False
    test()
