from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file="gp_ard.model.stan")
fit = model.sample(data="gp_ard.data.json", output_dir="./", show_console=True)
print(fit)
print(fit.summary())
print(fit.diagnose())
# fit.save_csvfiles()
fit.draws_pd().to_csv('samples_gp.csv',index=False)