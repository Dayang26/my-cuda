import config


def run():
    # 创建训练数据集和开发数据集
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)


if __name__ == "__main__":
    pass
