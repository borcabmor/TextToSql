import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb


def evaluate_model(model, dataset, device, batch_size=16):
    validation_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in validation_loader:
            q_ids = batch["question_input_ids"].to(device)
            q_mask = batch["question_attention_mask"].to(device)

            sql_ids = batch["sql_input_ids"].to(device)
            sql_mask = batch["sql_attention_mask"].to(device)

            q_emb, sql_emb = model(q_ids, q_mask, sql_ids, sql_mask)

            logits = q_emb @ sql_emb.T
            labels = torch.arange(logits.size(0), device=device)

            loss = F.cross_entropy(logits, labels)

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / len(validation_loader), total_acc / len(validation_loader)


def train_model(
    model,
    train_dataset,
    val_dataset,
    lr=2e-5,
    epochs=10,
    weight_decay=0.01,
    batch_size=16,
    device="cpu",
    patience=3,
):
    logger = logging.getLogger(__name__)

    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    best_val_acc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # --- Training ---
        # Set model in train mode
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            q_ids = batch["question_input_ids"].to(device)
            q_mask = batch["question_attention_mask"].to(device)
            sql_ids = batch["sql_input_ids"].to(device)
            sql_mask = batch["sql_attention_mask"].to(device)

            optimizer.zero_grad()

            # Call forward
            q_emb, sql_emb = model(q_ids, q_mask, sql_ids, sql_mask)

            # Contrastive loss: question i should match SQL i
            logits = q_emb @ sql_emb.T
            labels = torch.arange(logits.size(0), device=device)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        val_loss, val_acc = evaluate_model(model, val_dataset, device, batch_size)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # --- Early stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")

                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Best model weights restored")

    return model
