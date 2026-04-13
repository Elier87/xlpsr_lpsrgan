# Training Flow

```mermaid
flowchart LR
    subgraph D0[Stage 0 Offline Degradation]
        UFPRHR[UFPR HR images]
        NRCD[n-stage random combination degradation]
        SYN[Synthetic LR images]
        UFPRHR --> NRCD
        NRCD --> SYN
    end

    subgraph D1[Stage 1 Paired Data]
        UFPR[UFPR-SR-Plates paired LR-HR]
        LR1[LR crop]
        HR1[HR crop]
        GT1[GT plate text]
        UFPR --> LR1
        UFPR --> HR1
        UFPR --> GT1
    end

    subgraph D2[Stage 3 Challenge Data]
        XLPSR[XLPSR development set 39 sequences]
        SEQ[Sequence frames]
        DET[detections.json / bbox]
        GT3[GT plate text]
        XLPSR --> SEQ
        XLPSR --> DET
        XLPSR --> GT3
    end

    subgraph SR[LPSRGAN SR Backbone]
        G[LPSRGAN Generator G]
        D[LPSRGAN Discriminator D]
    end

    subgraph OCR[OCR Branch]
        P[PARSeq backbone frozen]
        FH[FRLP head trainable in Stage 3]
    end

    subgraph S1[Stage 1 Paired SR Pretrain / GAN]
        LR1 --> G
        G --> SR1[SR image]
        SR1 --> PIX[pixel loss]
        SR1 --> PER[perceptual loss]
        SR1 --> ADV_G[generator adversarial loss]
        HR1 --> PIX
        HR1 --> PER
        SR1 --> D
        HR1 --> D
        D --> ADV_D[discriminator loss]
    end

    subgraph S2[Stage 2 OCR-aware Paired Training]
        SR1 --> P
        HR1 --> P
        LR1 --> P
        P --> OCRVAL[OCR evaluate pred / conf / score]
        P --> OCRSUP[optional OCR-aware supervision]
    end

    subgraph S3[Stage 3 Challenge Finetune]
        SEQ --> SEL[frame selection]
        DET --> CROP[bbox crop / normalize]
        SEL --> CROP
        CROP --> LR3[sequence LR crops]
        LR3 --> G
        G --> SR3[SR sequence frames]
        SR3 --> P
        P --> FH
        FH --> TOK[token logits]
        FH --> TYPE[position-type logits]
        FH --> EOS[length / EOS structure]

        GT3 --> TCE[token CE]
        TOK --> TCE

        GT3 --> PCE[position-type CE]
        TYPE --> PCE

        GT3 --> LCE[length / EOS loss]
        EOS --> LCE

        TOK --> DEC[FRLP decode threshold + restricted vocab]
        DEC --> SCORE[score / acc / char acc]
    end

    subgraph OUT[Outputs]
        CKPT[checkpoint]
        LOG[log.txt / terminal]
        TB[tensorboard]
        CSV[val_ocr epoch csv]
        IMG[val images LR / SR / pred / score]
    end

    SYN --> UFPR
    PIX --> CKPT
    PER --> CKPT
    ADV_G --> CKPT
    ADV_D --> CKPT
    OCRVAL --> LOG
    OCRVAL --> TB
    OCRSUP --> CKPT
    TCE --> CKPT
    PCE --> CKPT
    LCE --> CKPT
    SCORE --> LOG
    SCORE --> TB
    SCORE --> CSV
    SR3 --> IMG
    DEC --> IMG
```
## Stage 0

Stage 0 是離線資料生成，不進入 training loop。本階段以 UFPR 的 HR 影像為輸入，透過 nRCD 隨機多階段劣化產生 synthetic LR，並保留原本資料夾結構與 HR/metadata，只重建 `lr-*.png`。目的不是訓練模型，而是先做可控的資料擴充，讓 Stage 1 能在更接近真實監視器退化的 paired 條件下預訓練。

## Stage 1

Stage 1 使用 UFPR-SR-Plates 的 paired LR-HR 資料做基礎 SR 訓練。第一步是 generator pretrain，主要靠 pixel 與 perceptual loss 把重建能力學穩；第二步再加入 discriminator 進行 paired GAN training，提升紋理與視覺細節。這一階段的核心目標是得到穩定的 SR backbone，並維持可正常 checkpoint、resume、validation 的訓練流程。

## Stage 2

Stage 2 保留 Stage 1 的 paired SR 主體，另外接入 frozen PARSeq 作為 external OCR teacher / evaluator。這一階段不重訓 OCR backbone，而是利用 OCR prediction、confidence、score 來做 OCR-aware validation，並視設定加入保守的 OCR supervision，讓 SR 結果不只看重建，也更朝向可辨識文字優化。

## Stage 3

Stage 3 使用 XLPSR development set 的 sequence 資料做 challenge-specific finetune。流程包含讀取 sequence frame、依 detections.json 做 bbox crop、選幀後送進 generator 產生 SR，再接 frozen PARSeq backbone 與 trainable FRLP head。主要 supervision 改成 token、position type、length/EOS 等法國牌照任務導向 loss，並輸出 score、CSV 與可視化結果。


