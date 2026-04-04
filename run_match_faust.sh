mesh1=002
mesh2=011
mesh2=020
mesh2=034
mesh2=098
# for mesh2 in {011,020,034,098}
for mesh2 in {061,}
do
python scripts/matcher.py --mesh1 tmp_FAUST/tr_reg_${mesh1}.obj  --mesh2 tmp_FAUST/tr_reg_${mesh2}.obj  \
    --feat1 tmp_teaser/faust/tr_reg_${mesh1}/inferred/net_evec.npy \
    --feat2 tmp_teaser/faust/tr_reg_${mesh2}/inferred/net_evec.npy \
    --signature=HKS --wks_num_E=30 --k_process=50 --out_dir tmp_out_corr/${mesh1}_${mesh2}\
    --zo_k_init 20 --zo_step 4 --zo_nit 5
done
