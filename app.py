from flask import Flask, request, render_template, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model (assuming you already uploaded generator.pth)
class UNetGenerator(torch.nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder = torch.nn.Sequential(
            self._block(1, 64, 4, 2, 1),
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            self._block(512, 512, 4, 2, 1),
        )

        self.decoder = torch.nn.Sequential(
            self._upblock(512, 512, 4, 2, 1),
            self._upblock(512, 256, 4, 2, 1),
            self._upblock(256, 128, 4, 2, 1),
            self._upblock(128, 64, 4, 2, 1),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1),
            torch.nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2)
        )

    def _upblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize and load weights
generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load("GGenerator.pth", map_location=device))
generator.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    image = Image.open(file).convert('L')  # Convert to grayscale
    sar_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = generator(sar_tensor).squeeze(0).cpu()

    output_image = transforms.ToPILImage()(output)
    
    img_io = io.BytesIO()
    output_image.save(img_io, format='JPEG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
