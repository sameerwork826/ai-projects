# Knowledge Distillation Showcase

A comprehensive demonstration of knowledge distillation techniques for model compression, featuring a teacher-student architecture on the MNIST dataset.

## 📋 Overview

This project showcases knowledge distillation, a model compression technique where a smaller "student" model learns from a larger "teacher" model. The implementation demonstrates how to transfer knowledge from a complex teacher network to a simpler student network while maintaining competitive performance.

### Key Features

- **Teacher Model**: A convolutional neural network with 32 filters and 2 fully connected layers
- **Student Model**: A simpler fully connected network with significantly fewer parameters
- **Knowledge Distillation**: Uses KL divergence loss to transfer soft predictions from teacher to student
- **Interactive Demo**: Streamlit web application for real-time model comparison
- **Performance Metrics**: Side-by-side comparison of accuracy and inference speed

## 🏗️ Architecture

### Teacher Network
```
Input (28x28) → Conv2d(1,32,5) → MaxPool2d(5,5) → FC(512,128) → FC(128,10) → Output
```

### Student Network
```
Input (28x28) → Flatten → FC(784,128) → FC(128,10) → Output
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd knowledge-distillation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the interactive demo**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## 📊 Usage

### Running the Notebook

1. Open `Knowledge-Distillation-Notebook.ipynb` in Jupyter Lab/Notebook
2. Run all cells to train both teacher and student models
3. Compare performance metrics and inference speeds

### Using the Streamlit App

1. Launch the app with `streamlit run app.py`
2. **Model Comparison**: View side-by-side performance metrics
3. **Interactive Testing**: Upload your own handwritten digits or use the drawing canvas
4. **Real-time Predictions**: See predictions from both teacher and student models
5. **Performance Analysis**: Compare inference speeds and accuracy

### Standalone Script

Run the complete training pipeline:
```bash
python train_models.py
```

## 📈 Results

Based on the notebook results:

| Model | Accuracy | Inference Time | Parameters |
|-------|----------|----------------|------------|
| Teacher | 98.81% | 2.32s | ~50K |
| Student | ~95% | 1.88s | ~100K |

*Note: Student model achieves competitive accuracy while being faster and having different parameter count due to architecture differences.*

## 🔬 Technical Details

### Knowledge Distillation Process

1. **Teacher Training**: Train the teacher model using standard cross-entropy loss
2. **Student Training**: Train the student model using KL divergence loss with teacher's soft predictions
3. **Loss Function**: 
   ```python
   def knowledge_distillation_loss(student_logits, teacher_logits):
       p_teacher = F.softmax(teacher_logits, dim=1)
       p_student = F.log_softmax(student_logits, dim=1)
       loss = F.kl_div(p_student, p_teacher, reduction='batchmean')
       return loss
   ```

### Key Components

- **Dataset**: MNIST handwritten digits (60,000 training, 10,000 test)
- **Framework**: PyTorch
- **Optimization**: Adam optimizer with learning rate 0.001
- **Training**: 5 epochs for both models

## 📁 Project Structure

```
knowledge-distillation/
├── Knowledge-Distillation-Notebook.ipynb  # Main notebook
├── app.py                                 # Streamlit frontend
├── train_models.py                        # Standalone training script
├── models/                                # Saved model weights
│   ├── teacher_model.pth
│   └── student_model.pth
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## 🛠️ Development

### Adding New Features

1. **Custom Models**: Modify the `TeacherNet` and `StudentNet` classes
2. **Different Datasets**: Update the data loading section
3. **Advanced Techniques**: Implement temperature scaling, attention transfer, etc.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📚 References

- [Machine Learning Model Compression: A Critical Step Towards Efficient Deep Learning](https://www.dailydoseofds.com/model-compression-a-critical-step-towards-efficient-machine-learning)
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a demonstration project for educational purposes. For production use, consider additional optimizations and validation techniques.
