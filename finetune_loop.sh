systemctl --user stop kairos-idle-finetune.timer
set -a && source ~/.config/kairos/kairos.env && set +a
count=1
until [ $count -gt 500 ]; do
  echo -ne "\033]0;Finetuning job: Loop # $count\007"
  echo "##### fintetune Count: $count #####"
  ((count++))
  uv run ./strategy/kairos_pipeline.py --stage finetune_next
done
systemctl --user start kairos-idle-finetune.timer
