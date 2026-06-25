"""
Script para descargar BloodMNIST y guardar como imágenes PNG organizadas por carpetas.

Estructura generada:
  data/
    train/
      0/  (Basophil)
      1/  (Eosinophil)
      ...
      7/  (Platelet)
    val/
      0/ ... 7/
    test/
      0/ ... 7/

Cada imagen se guarda como PNG de 28x28 RGB.
"""

import numpy as np
import os

CLASS_NAMES = [
    "Basophil",
    "Eosinophil", 
    "Erythroblast",
    "Immature_Granulocytes",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet"
]

def download_npz():
    """Descarga el archivo .npz de BloodMNIST desde Zenodo."""
    url = "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=true"
    output = "bloodmnist.npz"

    if os.path.exists(output):
        print(f"✓ {output} ya existe, saltando descarga.")
        return output

    print(f"Descargando BloodMNIST desde Zenodo...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, output)
        print(f"✓ Descarga completa: {output}")
    except Exception as e:
        print(f"✗ Error descargando: {e}")
        raise

    return output

def save_images(images, labels, split_name, base_dir="data"):
    """
    Guarda imágenes como PNG organizadas en carpetas por clase.
    images: (N, 28, 28, 3) uint8
    labels: (N, 1) int
    """
    from PIL import Image

    n = len(images)
    split_dir = os.path.join(base_dir, split_name)

    # Crear carpetas por clase
    for cls in range(8):
        os.makedirs(os.path.join(split_dir, str(cls)), exist_ok=True)

    # Contador por clase para nombrar archivos
    class_count = [0] * 8

    for i in range(n):
        label = int(labels[i].item() if hasattr(labels[i], 'item') else labels[i])
        img = Image.fromarray(images[i])  # (28, 28, 3) RGB
        
        filename = f"{class_count[label]:05d}.png"
        filepath = os.path.join(split_dir, str(label), filename)
        img.save(filepath)
        class_count[label] += 1

        if (i + 1) % 2000 == 0 or i == n - 1:
            print(f"  {split_name}: {i+1}/{n} imágenes guardadas")

    print(f"✓ {split_name}/ completado ({n} imágenes)")
    for cls in range(8):
        print(f"    Clase {cls} ({CLASS_NAMES[cls]}): {class_count[cls]} imágenes")

def main():
    npz_file = download_npz()

    print("\nCargando datos...")
    data = np.load(npz_file)

    print(f"  Train: {data['train_images'].shape}")
    print(f"  Val:   {data['val_images'].shape}")
    print(f"  Test:  {data['test_images'].shape}")

    all_labels = np.concatenate([data['train_labels'], data['val_labels'], data['test_labels']])
    num_classes = len(np.unique(all_labels))
    print(f"  Clases: {num_classes}")

    print("\nGuardando imágenes PNG...")
    save_images(data['train_images'], data['train_labels'], "train")
    save_images(data['val_images'], data['val_labels'], "val")
    save_images(data['test_images'], data['test_labels'], "test")

    print("\n============================")
    print("  Conversión completada ✓")
    print("============================")
    print(f"\nEstructura: data/{{train,val,test}}/{{0..7}}/XXXXX.png")
    print(f"  Imágenes: 28x28 RGB, 8 clases")
    print(f"  Train: {data['train_images'].shape[0]}")
    print(f"  Val:   {data['val_images'].shape[0]}")
    print(f"  Test:  {data['test_images'].shape[0]}")

if __name__ == "__main__":
    main()