# TRA3: a TRAnsformer-Based TRAditional Chinese TRAslation Model
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<img src="image\logo.jpg" width="300" />

This is the implementation of TRA3, a TRAnsformer-Based TRAditional Chinese TRAslation Model. TRA3 is a model that translates between Traditional Chinese and Simplified Chinese. It is based on the Transformer architecture and is trained on the a subset of the [Classical Chinese](https://github.com/BangBOOM/Classical-Chinese/tree/master) dataset.

## 🔍 Table of Contents

- [📥 Installation](#installation)
- [🚀 Usage](#usage)
- [🤝 Contributing](#contributing)
- [📝 License](#license)

## 📥 Installation

Instructions on how to install and run your project.

```bash
$ git clone https://github.com/AnonChihayaaa/transformer_classic2chinese.git
$ cd transformer_classic2chinese
$ pip install -r requirements.txt
```
Or alternatively, you can simply run the following command and manually install the required packages.
```bash
$ python train.py
```

## 🚀 Usage
### 📚 Training
To train the model, run the following command:
```bash
$ python train.py
```
After training, the model will be saved in the `models` directory.

### 📊 Evaluation
To evaluate the model, run the following command:
```bash
$ python evaluate.py
```

### ⚙️ Changing the training parameters
If you want to change the training parameters, you can do so by modifying the `utils.py` file. The following parameters can be changed:
- `BATCH_SIZE`: The batch size used during training. The default value is 128.
- `EPOCHS`: The number of epochs to train the model. The default value is 100.
- `LAYERS`: The number of layers in the Transformer model. The default value is 6.
- `H_NUM`: The number of attention heads in the Transformer model. The default value is 8.
- `D_MODEL`: The dimension of the model. The default value is 256.
- `D_FF`: The dimension of the feedforward layer. The default value is 1024.
- `DROPOUT`: The dropout rate. The default value is 0.1.
- `MAX_LENGTH`: The maximum length of the input sequence. The default value is 120.

### 📂 Changing the dataset
If you want to train the model on a different dataset, you can do so by modifying the dataset under `nmt\Classic-Chinese\` directory. The dataset should be in the following format:
```
<Traditional Chinese Sentence>\t<Simplified Chinese Sentence>
<Traditional Chinese Sentence>\t<Simplified Chinese Sentence>
...
```

## 🤝 Contributing

Guidelines on how to contribute to your project.

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

You can contact me at <vincentzhao1024@gmail.com>.

## 📝 License

This project is licensed under the [MIT License](LICENSE).
