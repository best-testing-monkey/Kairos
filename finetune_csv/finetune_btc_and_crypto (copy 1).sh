echo "###########" 
echo "### R 1 ###" 
echo "###########" 

# r1 (~1.5h) — BTC only, from pretrained Kronos-base:
cd ..
uv run examples/run_backtest_kairos.py --model finetune_csv/models/btc_base_finetune/basemodel/best_model --output output/BTC-USD_backtest_results_finetuned_btc.png

echo "###########" 
echo "### R 2 ###" 
echo "###########" 

# r2 (~1.5h) — BTC only, from r1, lower LR:
uv run examples/run_backtest_kairos.py --model finetune_csv/models/btc_base_finetune_r2/basemodel/best_model --output output/BTC-USD_backtest_results_finetuned_btc_extra.png

echo "############" 
echo "### R 3* ###" 
echo "############" 

# r3-continued (optional, repeatable) — same 22 assets, in-place refinement:
uv run examples/run_backtest_kairos.py --model finetune_csv/models/btc_base_finetune_r3/basemodel/best_model --output output/BTC-USD_backtest_results_finetuned_crypto_extra.png

cd finetune_csv
