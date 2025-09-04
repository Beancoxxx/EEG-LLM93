"""
TimeLLM 模型修改版本 - 支持EEG情绪分类
保持向后兼容，最小改动原则
"""

import torch
import torch.nn as nn
from math import sqrt
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
import transformers

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    """原始的预测头，保持不变"""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ClassificationHead(nn.Module):
    """新增：分类头，用于EEG情绪分类"""
    def __init__(self, n_vars, d_model, patch_nums, num_class, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.patch_nums = patch_nums
        self.num_class = num_class
        
        # 方案1：最简单的平均池化 + 线性层
        # 输入: [batch, n_vars, d_model, patch_nums]
        # 目标: [batch, num_class]
        
        # 计算输入特征维度
        self.input_dim = n_vars * d_model
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.input_dim // 2, num_class)
        )
        
        print(f"[ClassificationHead] 初始化完成:")
        print(f"  - 输入维度: {self.input_dim} (n_vars={n_vars} × d_model={d_model})")
        print(f"  - 输出类别: {num_class}")
        
    def forward(self, x):
        """
        输入: x shape [batch, n_vars, d_model, patch_nums]
        输出: [batch, num_class]
        """
        batch_size = x.shape[0]
        
        # 打印输入维度（调试用）
        # print(f"[ClassificationHead] 输入形状: {x.shape}")
        
        # Step 1: 对所有patch取平均 (最简单的聚合方式)
        # [batch, n_vars, d_model, patch_nums] -> [batch, n_vars, d_model]
        x = torch.mean(x, dim=-1)
        
        # Step 2: 展平通道维度
        # [batch, n_vars, d_model] -> [batch, n_vars * d_model]
        x = x.reshape(batch_size, -1)
        
        # Step 3: 分类
        # [batch, n_vars * d_model] -> [batch, num_class]
        x = self.classifier(x)
        
        return x


class Model(nn.Module):
    """修改后的TimeLLM模型，支持分类任务"""
    
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        # 打印任务信息
        print(f"\n[TimeLLM] 初始化模型 - 任务类型: {self.task_name}")
        
        # LLM配置（保持原样）
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("下载Llama模型...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
                
            self.tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
            )
            
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token


        elif configs.llm_model == 'GPT2':
            from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
            
            # 修改为本地路径
            gpt2_model_id = '/home/lzb/wby/New_Project/TimeLLM/gpt2'
            
            # 从本地加载配置
            self.gpt2_config = GPT2Config.from_pretrained(gpt2_model_id, local_files_only=True)
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_hidden_states = True
            self.gpt2_config.output_attentions = True
            
            # 从本地加载模型
            self.llm_model = GPT2Model.from_pretrained(
                gpt2_model_id,
                config=self.gpt2_config,
                local_files_only=True
            )
            
            # 从本地加载分词器
            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_id, local_files_only=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 冻结LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        # 任务描述（根据任务类型调整）
        if self.task_name == 'classification':
            self.description = 'EEG signals for emotion recognition'
        else:
            self.description = 'Time series forecasting'
            
        self.dropout = nn.Dropout(configs.dropout)
        
        # Patch嵌入层（保持原样）
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        
        # 词嵌入和重编程层（保持原样）
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        
        # 计算patch数量
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums
        
        # 根据任务类型选择输出投影层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(
                configs.enc_in, self.head_nf, self.pred_len,
                head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            # 新增：分类头
            self.output_projection = ClassificationHead(
                n_vars=configs.enc_in,
                d_model=self.d_ff,
                patch_nums=self.patch_nums,
                num_class=configs.num_class,
                dropout=configs.dropout
            )
            print(f"[TimeLLM] 分类任务配置:")
            print(f"  - 类别数: {configs.num_class}")
            print(f"  - 通道数: {configs.enc_in}")
            print(f"  - Patch数: {self.patch_nums}")
        else:
            raise NotImplementedError(f"不支持的任务类型: {self.task_name}")
        
        # 标准化层
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        统一的forward接口，根据任务类型调用不同的处理流程
        
        对于分类任务:
        - x_enc: [batch_size, seq_len, n_channels] 来自data_loader_eeg
        - 返回: [batch_size, num_class] 分类logits
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            # 分类任务
            return self.classification(x_enc, x_mark_enc)
        else:
            raise ValueError(f"未知的任务类型: {self.task_name}")
    
    def classification(self, x_enc, x_mark_enc):
        """
        分类任务的前向传播
        复用大部分forecast的代码，只修改必要部分
        """
        # 打印输入维度（调试用）
        # print(f"[classification] 输入 x_enc 形状: {x_enc.shape}")
        
        # Step 1: 标准化
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        # Step 2: 重塑数据（与forecast相同）
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Step 3: 计算统计信息（简化版提示词）
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        
        # Step 4: 生成提示词（基于DEAP.txt的专业提示词）
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            
            # 基于DEAP.txt内容的专业提示词
            if N == 32:  # DEAP数据集（32通道）
                prompt_ = (
                    f"<|start_prompt|>Dataset Description: The DEAP dataset contains 32-channel EEG signals recorded from participants experiencing emotional stimuli through music videos. Participants rated each video on continuous scales of valence (negative to positive) and arousal (calm to excited)."
                    f"Task Instruction: This is a binary emotion classification task. Analyze the EEG signal patterns to determine whether the participant is experiencing positive (high valence) or negative (low valence) emotional states."
                    f"EEG Channel Configuration: 32 channels following the International 10-20 system. Frontal channels (Fp1, Fp2, F3, F4, F7, F8, Fz) associated with emotional processing and executive control. Central channels (C3, C4, Cz) related to sensorimotor integration. Parietal channels (P3, P4, P7, P8, Pz) involved in attention and spatial processing. Occipital channels (O1, O2) for visual processing. Temporal channels (T7, T8) for auditory and memory processing."
                    f"Emotion-Related EEG Patterns to analyze:"
                    f"Positive Emotions (High Valence): Increased left frontal alpha power (8-13 Hz) relative to right frontal, enhanced gamma band activity (30-45 Hz) in frontal regions, higher beta power (13-30 Hz) indicating active cognitive engagement, greater coherence between frontal and parietal regions."
                    f"Negative Emotions (Low Valence): Increased right frontal alpha activity, reduced overall gamma power, enhanced theta activity (4-8 Hz) in frontal and central regions, lower beta/alpha ratio, increased delta power (0.5-4 Hz) in some cases."
                    f"Input statistics for current sample: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, signal trend {'increasing' if trends[b] > 0 else 'decreasing'}."
                    f"Key Features to Analyze: Power spectral density in each frequency band, frontal asymmetry patterns, inter-channel coherence, temporal dynamics and transitions, signal complexity and entropy."
                    f"Classification: Based on the 32-channel EEG signal patterns, classify whether the emotional state is positive (high valence) or negative (low valence).<|end_prompt|>"
                )
            elif N == 62:  # SEED数据集（62通道）
                prompt_ = (
                    f"<|start_prompt|>Dataset Description: The SEED dataset contains 62-channel EEG signals recorded during emotion elicitation using carefully selected film clips. The dataset targets three distinct emotional states: positive (happiness), neutral, and negative (sadness/fear)."
                    f"Task Instruction: This is a three-class emotion classification task. Analyze the comprehensive EEG signal patterns to classify the participant's emotional state into one of three categories: positive, neutral, or negative."
                    f"EEG Channel Configuration: 62 channels with extended 10-20 montage providing high spatial resolution. Prefrontal channels (Fp1, Fp2, AFz, AF3, AF4) for executive control and emotion regulation. Frontal channels (F1-F8, Fz, FCz) for emotional processing and motor planning. Central channels (C1-C6, Cz, CPz) for sensorimotor integration. Parietal channels (P1-P8, Pz, POz) for attention and spatial awareness. Occipital channels (O1, O2, Oz) for visual processing. Temporal channels (T7, T8, TP7, TP8, FT7, FT8) for auditory and memory."
                    f"Emotion-Specific EEG Signatures:"
                    f"Positive Emotions: Significantly increased gamma power (30-50 Hz) in left temporal and frontal regions, enhanced beta activity (14-30 Hz) across frontal cortex, characteristic frontal alpha asymmetry (left < right), high-frequency oscillations in temporal-parietal networks, increased connectivity between frontal and temporal regions."
                    f"Neutral State: Balanced alpha power (8-13 Hz) between hemispheres, moderate beta activity without significant lateralization, lower gamma power compared to emotional states, stable theta rhythms (4-8 Hz) in frontal-central areas, reduced long-range connectivity."
                    f"Negative Emotions: Increased right frontal and temporal alpha power, enhanced theta activity in frontal and central regions, reduced gamma power especially in left hemisphere, increased delta power (1-4 Hz) in frontal areas, greater alpha-theta coupling, heightened connectivity in right hemisphere networks."
                    f"Input statistics for current sample: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, signal trend {'increasing' if trends[b] > 0 else 'decreasing'}."
                    f"Critical Analysis Points: Differential entropy features across bands, hemispheric power asymmetry, phase-locking values between regions, spectral power distribution, temporal evolution of emotional patterns, cross-frequency coupling."
                    f"Classification: Based on the 62-channel EEG signal patterns, classify the emotional state into positive, neutral, or negative categories.<|end_prompt|>"
                )
            else:  # 通用提示词
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description};"
                    f"Task description: classify the emotional state based on EEG signals;"
                    f"Input statistics: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, signal trend {'increasing' if trends[b] > 0 else 'decreasing'};"
                    f"Analyze the multi-channel EEG signal patterns to determine the emotional state classification.<|end_prompt|>"
                )
            prompt.append(prompt_)
        
        # 其余代码保持不变...
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        # 获取提示词嵌入
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, 
                              truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))
        
        # 其余处理流程保持不变...
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.float())
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        
        dec_out = dec_out[:, :, :, -self.patch_nums:]
        
        output = self.output_projection(dec_out)
        
        return output
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """原始的预测函数，保持不变"""
        # [原始代码保持不变...]
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.float())
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out
    
    def calcute_lags(self, x_enc):
        """计算滞后相关性，保持原样"""
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    """重编程层，保持原样"""
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("测试修改后的TimeLLM模型")
    print("=" * 70)
    
    # 创建测试配置
    class TestConfig:
        def __init__(self, task='classification'):
            # 基础配置
            self.task_name = task
            self.seq_len = 256
            self.pred_len = 96
            self.label_len = 48
            
            # 模型配置
            self.d_model = 32
            self.n_heads = 8
            self.d_ff = 128
            self.dropout = 0.1
            self.patch_len = 16
            self.stride = 8
            
            # LLM配置
            self.llm_model = 'LLAMA'
            self.llm_dim = 4096
            self.llm_layers = 8
            
            # 数据配置
            if task == 'classification':
                self.enc_in = 32  # DEAP的32个通道
                self.num_class = 2  # 二分类
            else:
                self.enc_in = 7  # 预测任务的通道数
    
    # 测试分类任务
    print("\n1. 测试分类任务")
    config = TestConfig('classification')
    model = Model(config)
    
    # 创建随机输入（模拟data_loader_eeg的输出）
    batch_size = 4
    x = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark = torch.zeros(batch_size, config.seq_len, 4)
    
    print(f"\n输入数据形状:")
    print(f"  - x: {x.shape}")
    print(f"  - x_mark: {x_mark.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x, x_mark)
    
    print(f"\n输出形状: {output.shape}")
    print(f"期望形状: [{batch_size}, {config.num_class}]")
    print(f"输出样例: {output[0].numpy()}")
    
    # 测试预测任务（确保向后兼容）
    print("\n" + "=" * 50)
    print("2. 测试预测任务（向后兼容）")
    config = TestConfig('long_term_forecast')
    model = Model(config)
    
    # 创建预测任务的输入
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.zeros(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.enc_in)
    x_mark_dec = torch.zeros(batch_size, config.label_len + config.pred_len, 4)
    
    print(f"\n预测任务输入形状:")
    print(f"  - x_enc: {x_enc.shape}")
    print(f"  - x_dec: {x_dec.shape}")
    
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\n预测输出形状: {output.shape}")
    print(f"期望形状: [{batch_size}, {config.pred_len}, {config.enc_in}]")
