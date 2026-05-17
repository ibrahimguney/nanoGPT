# nanoGPT Çalışma Notları

Bu repo, Andrej Karpathy'nin [nanoGPT](https://github.com/karpathy/nanoGPT) projesini kullanarak transformer mimarisini sıfırdan öğrenme yolculuğumun kaydıdır.

## Ne öğrendim?

- Transformer mimarisinin tamamı (LayerNorm, Attention, MLP, Block, GPT)
- Q/K/V matematiği, causal masking, multi-head
- Embedding, position embedding, weight tying
- Autoregressive generation, temperature, top-k sampling
- PyTorch temelleri (nn.Module, forward, parameters)
- Cross entropy ile eğitim

## Çalışma notu

Detaylı notlar burada:
- 📕 [PDF versiyonu](./nanoGPT-calisma-notu.pdf)
- 📝 [Markdown versiyonu](./nanoGPT-calisma-notu.md)

## Kendi modelimi eğittim!

Shakespeare verisi üzerinde küçük bir GPT eğittim (CPU, ~1M parametre, 2000 adım).

Loss 4.2'den 1.77'ye düştü. Model Shakespeare oyun formatını, karakter isimlerini, İngilizce yazımını **sıfırdan** öğrendi.

## Nasıl çalıştırılır?

```bash
# Veriyi hazırla
python data/shakespeare_char/prepare.py

# Eğit (CPU için küçük model)
python train.py config/train_shakespeare_char.py --device=cpu --compile=False ^
    --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 ^
    --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 ^
    --lr_decay_iters=2000 --dropout=0.0

# Örnekle
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

## Kaynaklar

- [Karpathy / nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

ÇIKTI: # Veriyi hazırla
python data/shakespeare_char/prepare.py

# Eğit (CPU için küçük model)
python train.py config/train_shakespeare_char.py --device=cpu --compile=False ^
    --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 ^
    --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 ^
    --lr_decay_iters=2000 --dropout=0.0

# Örnekle
python sample.py --out_dir=out-shakespeare-char --device=cpu

<img width="989" height="1325" alt="image" src="https://github.com/user-attachments/assets/3e4ec38a-671c-4363-9962-a8e741ac413f" />
