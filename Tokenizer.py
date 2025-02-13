from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder 

tokenizer = Tokenizer(BPE(
    unk_token="[UNK]",
    continuing_subword_prefix="##" 
))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()  
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A <|endoftext|>",
    special_tokens=[
        ("[BOS]", 1),
        ("<|endoftext|>", 2),
    ]
)

trainer = BpeTrainer(
    vocab_size=12000,
    min_frequency=1,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "<|endoftext|>"],
    continuing_subword_prefix="##"  
)

# Train and save the tokenizer:

#tokenizer.train(files=["Data_small.txt"], trainer=trainer)
#tokenizer.save("tokenizer.json")
