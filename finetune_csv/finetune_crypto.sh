echo "###########" 
echo "### R 3 ###" 
echo "###########" 

# r3 (~10h) — 22 crypto assets, from r2:
uv run finetune_base_model.py --config configs/finetune_btc_base_r3.yaml
cd ..
uv run examples/run_backtest_kairos.py --model finetune_csv/models/btc_base_finetune_r3/basemodel/best_model --output output/BTC-USD_backtest_results_finetuned_crypto.png
