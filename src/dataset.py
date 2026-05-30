from torch.utils.data import Dataset


class SpiderDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.questions = df["question"].tolist()
        self.queries = df["query"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q_enc = self.tokenizer(
            self.questions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        sql_enc = self.tokenizer(
            self.queries[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "question_input_ids": q_enc["input_ids"].squeeze(0),
            "question_attention_mask": q_enc["attention_mask"].squeeze(0),
            "sql_input_ids": sql_enc["input_ids"].squeeze(0),
            "sql_attention_mask": sql_enc["attention_mask"].squeeze(0),
        }
