# Hateful memes fine tuning dynamics 
## Supports sampling for DPO 


## DPO training 
- Try implement with Llamafactory and trl
- Sampling with same prompt as in the grpo training (without structured reasoning)
- With the correct, chosen response generated in dpo sampling, use these for SFT as well. 






## Analysis 
### Comparing the KL divergence
- Plot out the KL divergence for SFT, DPO and GRPO models for in domain and out of domain (train set and val set and other dataset)
	- KL divergence: defined as the KL divergence between the instruct model and the trained model
### Likelihood of chosen and rejected likelihood on val set. 
- Comparing the chosen and rejected samples likelihood on the val set. 
	- These samples are those generated with the reference model (initial instruct model)

### Comparing different reward
- Check how well different model keeps the format 
- Check how the different reward increase for the different training.


### Check the rank of the weight difference
- Compute the $\delta W$ for each of the models trained with the reference model. 
- Plot the singular values, see if sft only picks up some of the features of the dataset, (with lower number of high singular values) while, grpo/dpo may have more high magnitude singular values 

### Sharpe feature inpormatance ratio 
- Check the feature inportance ratio with sharpe ratio analysis on the trrianing set. 
- This is to check overfitting, giving a good guidance to how a model is well trained.  


### Attention scores 
- Check the attention scores of the last input query token with all the previous input sequence, performing some form of aggregation (layer wise / token wise) 
- Or maybe add noise to the input image, and check how the attention scores changes to check for robustness to adversarial noise / generalization across domain / overfitting. 
- This is sdimilar to check overfitting, and how robust the model is 

- Perform this analysis for different training mechanism and for both the training set and test set. 
	- Help understand why the acc matched very well on the training set and test set on the GRPO trained model so well. 
	- Check if GRPO is better than DPO or not. 

### Entropy with sampling 
- Sampling with the fine-tuned model and reference model. 
- Treat hateful/benign as 1/0, define the binary entropy as the generated sequences difference in prediction. 
- Check for mutual information as well. 
- Check if the better model (higher accuracy model) has lower entropy, i.e., more confident, less confused.
	- If so, can we develop similar entropy minimization loss/mechanism that could train the models like GRPO?
	- Try to derive/connect the (one-shot) entropy minimization algorithm and GRPO/SFT loss. 