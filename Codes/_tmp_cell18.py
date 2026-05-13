original_phot = pd.DataFrame(SN.phot.copy())
extr_pts_pd = pd.DataFrame(columns=original_phot.columns)#.reindex_like(clipped_phot)[:0]

# Extrapolate BAND V first
#band_init = 'Bessell_V' if snname not in noBessellV_useswiftV else 'swift_V'
band_init = 'Swope_V' if snname not in noBessellV_useswiftV else 'Swope_V'
flux_V = SN.clipped_phot[SN.clipped_phot['band']==band_init]['Flux']
fluxerr_V = SN.clipped_phot[SN.clipped_phot['band']==band_init]['Flux_err']
t_V = SN.clipped_phot[SN.clipped_phot['band']==band_init]['MJD']

mjd_Vpeak = t_V[np.argmax(flux_V)]
Vpeak = max(flux_V)
if snname in pre_bump.keys():
    mjd_Vpeak = explosion_dates[snname][0]+pre_bump[snname][0]
    Vpeak = max(flux_V[t_V>min(t_V)+pre_bump[snname][0]])

plt.plot(t_V, flux_V/Vpeak, 'ok', alpha=0.5)
    
# window = 1.0  # days
# if np.count_nonzero(flux_V[t_V<=mjd_Vpeak]<=0.8*Vpeak)>=2:
#     max_MJD = max(t_V[t_V<=mjd_Vpeak][flux_V[t_V<=mjd_Vpeak]<=0.8*Vpeak])
# else:
#     max_MJD = mjd_Vpeak
#     #max_MJD = min(t_V) + window
# print ('V band', 'First data:%.2f'%(min(t_V)-mjd_Vpeak), 'Max point used for fitting:%.2f'%(max_MJD-mjd_Vpeak))

# t_ = t_V[t_V<=max_MJD]
# phase_ = t_ - mjd_Vpeak
# flux_ = flux_V[t_V<=max_MJD]/Vpeak
# fluxerr_ = fluxerr_V[t_V<=max_MJD]/Vpeak
# plt.vlines(max_MJD, 0,1, linestyle=':', alpha=0.5)

print ('V band', 'First data:%.2f'%(min(t_V)-mjd_Vpeak), 'Using entire lightcurve')

t_ = np.array(t_V)
phase_ = t_ - mjd_Vpeak
flux_ = np.array(flux_V)/Vpeak
fluxerr_ = np.array(fluxerr_V)/Vpeak

# FIT
R, cov, t_extrapV, fittedV, fittedV_err, t_newpts, newpts_, newpts_err, label_dict, success = performe_fit(snname, band_init, t_, flux_, fluxerr_, phase_, mjd_Vpeak)
    
plt.errorbar(t_, flux_, yerr=fluxerr_, fmt='ok', label='Data')
plt.ylabel('Flux (V band)', fontsize=15)
        
plt.plot(t_extrapV, fittedV, 'g--', lw=3, label='\n'.join("%s: %.1f"%(k,v) for (k,v) in label_dict))
plt.fill_between(t_extrapV, fittedV-fittedV_err, fittedV+fittedV_err,
                 facecolor='g', alpha=0.1)

plt.errorbar(t_newpts, newpts_, yerr=newpts_err, fmt='r.', ecolor='r', elinewidth=0.2, label='Extrap Data points')

plt.xlim(min(min(t_extrapV)-5., -25.+mjd_Vpeak),30.+mjd_Vpeak)
if snname in pre_bump.keys(): plt.xlim((explosion_dates[snname][0]-2.),20.+mjd_Vpeak)
    
if snname in se_sne: plt.title(snname+ '_SESN')
else: plt.title(snname+ ' Type II/IIn')
plt.ylim(-0.05, max(max(fittedV),1.1))
plt.xlim(57982, 57990)
plt.legend()
plt.show()
    
if success:
    extr_pts_pd['MJD'] = t_newpts
    extr_pts_pd['band'] = np.full(len(t_newpts), fill_value=band_init)
    extr_pts_pd['Flux'] = newpts_*Vpeak
    extr_pts_pd['Flux_err'] = newpts_err*Vpeak
    extr_pts_pd['FilterSet'] = np.full(len(t_newpts), fill_value='SUDO_PTS')
    extr_pts_pd['Instr'] = np.full(len(t_newpts), fill_value='SUDO_PTS')
    #original_phot = original_phot.append(extr_pts_pd)
    original_phot = pd.concat([original_phot, extr_pts_pd], ignore_index=True)
else: print ('Something went wrong')
    
i=1
early_bands=[]
exclude_bands = exclude_dict[snname] if snname in exclude_dict else []
include_bands = include_dict[snname] if snname in include_dict else []
#print(np.unique(SN.clipped_phot['band']))
# for band in np.unique(SN.clipped_phot['band']):
#     t_x = SN.clipped_phot[SN.clipped_phot['band']==band]['MJD']
#     #print(f"{band}: min(t_x)={min(t_x)}, min(t_V)={min(t_V)}, max_MJD={max_MJD}, t_x_before_maxMJD={t_x[t_x<=max_MJD]}")
#     #if (min(t_x)<=min(t_V)+2.0)&(len(t_x[t_x<=max_MJD])>=1):
#     #rav changed this line
#     # rav removing this line again 2/25/26, to add the new logic
#     # if (min(t_x)<=min(t_V)+2.0)&(len(t_x[t_x<=max(t_V)+1.0])>=1):

#     # NEW: Determine the absolute start date (fallback to first V observation if undefined)
#     start_date = explosion_dates[snname][0]
#     if start_date is None: 
#         start_date = min(t_V)

#     # Only include filters that have data within 1.5 days of the start date
#     if (min(t_x) - start_date <= 1.3):
#         if (snname in se_sne)&(band not in [band_init, 'swift_UVW1','swift_UVW2', 'swift_UVM2'])&(band not in exclude_bands):
#             early_bands.append(band)
#         elif (snname not in se_sne)&(band not in [band_init, 'swift_U', 'Bessell_U', 'swift_UVW1',
#                                            'swift_UVW2', 'swift_UVM2'])&(band not in exclude_bands):
#             early_bands.append(band)
#     if (band in include_bands):
#         if band not in early_bands: early_bands.append(band)

# N_rows = int(len(early_bands)/3)+1
# fig = plt.figure(figsize=(20,N_rows*5))
# print(early_bands)
# print ('Im extending also', early_bands, '\n')

# --- EXPLICIT FILTER SELECTION ---
# Define exactly which filters to fit. This bypasses the automated 1.3-day cutoff logic.
early_bands = ['DECam_i', 'DECam_z', 'Swope_i', 'FLAMINGOS-2_Ks', 'FourStar_H', 'FourStar_J', 'FourStar_Ks', 'GFC_i', 'GFC_y', 'GFC_z', 
                'HSC_z', 'SIRIUS_H', 'SIRIUS_J', 'SIRIUS_Ks', 'Sinistro_g', 'Sinistro_r', 'Skymapper_r', 'UVOT_M2', 'UVOT_U', 'UVOT_W1', 
                'VISTA_J', 'VISTA_Ks', 'VISTA_Y'] # <-- Modify this list with your filters of choice

N_rows = int(len(early_bands)/3)+1
fig = plt.figure(figsize=(20,N_rows*5))
print ('Im extending ONLY these explicitly chosen filters:', early_bands, '\n')

for band in early_bands:
    plt.subplot(N_rows,3,i)
    
    flux_x = SN.clipped_phot[SN.clipped_phot['band']==band]['Flux']
    fluxerr_x = SN.clipped_phot[SN.clipped_phot['band']==band]['Flux_err']
    t_x = SN.clipped_phot[SN.clipped_phot['band']==band]['MJD']
    xpeak = max(flux_x)
    mjd_xpeak = t_x[np.argmax(flux_x)]
    if snname in pre_bump.keys():
        mjd_xpeak = explosion_dates[snname][0]+pre_bump[snname][0]
        xpeak = max(flux_x[t_x>mjd_xpeak])
        
    plt.plot(t_x, flux_x/xpeak, 'ok', alpha=0.5)
    
    # Replacing this!!!
    # if np.count_nonzero(flux_x[t_x<=mjd_xpeak]<=0.8*xpeak)>=2:
    #     max_MJD = max(t_x[t_x<=mjd_xpeak][flux_x[t_x<=mjd_xpeak]<=0.8*xpeak])
    # else:
    #     max_MJD = mjd_xpeak
    # print (band, 'First data:%.2f'%(min(t_x)-mjd_xpeak), 'Max point used for fitting:%.2f'%(max_MJD-mjd_xpeak))

    # t_ = t_x[t_x<=max_MJD]
    # phase_ = t_ - mjd_xpeak
    # flux_ = flux_x[t_x<=max_MJD]/xpeak
    # fluxerr_ = fluxerr_x[t_x<=max_MJD]/xpeak
    # #plt.xlim(min(min(t_extrapV)-5., -25.+mjd_Vpeak),30.+mjd_Vpeak)
    # plt.xlim(57982, 57990)
    # if snname in pre_bump.keys(): plt.xlim((explosion_dates[snname][0]-2.),20.+mjd_Vpeak)
    # plt.vlines(max_MJD, 0,1, linestyle=':', alpha=0.5)

    #NEW!!!
    print (band, 'First data:%.2f'%(min(t_x)-mjd_xpeak), 'Using entire lightcurve')

    t_ = np.array(t_x)
    phase_ = t_ - mjd_xpeak
    flux_ = np.array(flux_x)/xpeak
    fluxerr_ = np.array(fluxerr_x)/xpeak
    
    plt.xlim(57982, 57990)
    if snname in pre_bump.keys(): plt.xlim((explosion_dates[snname][0]-2.),20.+mjd_Vpeak)

    results = performe_fit(snname, band, t_, flux_, fluxerr_, phase_, mjd_xpeak)
    R, cov, t_extrap, fitted, fitted_err, t_newpts, newpts_, newpts_err, label_dict, success= results
    
    plt.errorbar(t_, flux_, yerr=fluxerr_, fmt='ok', label='Data')
    
    plt.ylabel('Flux (%s band)'%band, fontsize=15)
    plt.plot(t_extrap, fitted, '--', color=color_dict[band], lw=2,
             label='\n'.join("%s: %.1f"%(k,v) for (k,v) in label_dict))
    plt.fill_between(t_extrap, fitted-fitted_err, fitted+fitted_err,
                 facecolor=color_dict[band], alpha=0.1)
    plt.errorbar(t_newpts, newpts_, yerr=newpts_err, fmt='r.', ecolor='r', elinewidth=0.2, label='Extrap Data points')
    plt.ylim(-0.05, max(max(fitted),1.1))
    
    if snname in se_sne: plt.title(snname+ '_SESN')
    else: plt.title(snname+ ' Type II/IIn')
    plt.legend()
    i=i+1
   
    if success:
        extr_pts_pd = pd.DataFrame(columns=original_phot.columns)#.reindex_like(clipped_phot)[:0]
        extr_pts_pd['MJD'] = t_newpts
        extr_pts_pd['band'] = np.full(len(t_newpts), fill_value=band)
        extr_pts_pd['Flux'] = newpts_ * xpeak
        extr_pts_pd['Flux_err'] = newpts_err * xpeak
        extr_pts_pd['FilterSet'] = np.full(len(t_newpts), fill_value='SUDO_PTS')
        extr_pts_pd['Instr'] = np.full(len(t_newpts), fill_value='SUDO_PTS')
        #original_phot = original_phot.append(extr_pts_pd) 
        original_phot = pd.concat([original_phot, extr_pts_pd], ignore_index=True)
    else: print ('Something went wrong')
    
    plt.subplot(N_rows,3,N_rows*3)
    plt.plot(t_extrap, fitted*xpeak, '-', color=color_dict[band], lw=3, label=band)
    plt.fill_between(t_extrap, (fitted-fitted_err)*xpeak, (fitted+fitted_err)*xpeak,
                         facecolor=color_dict[band], alpha=0.1)
    print ('\n')
plt.subplot(N_rows,3,N_rows*3)
plt.plot(t_extrapV, fittedV*Vpeak, '-', color=color_dict['Swope_V'], lw=3,
                 label='Swope_V')
plt.fill_between(t_extrapV, Vpeak*(fittedV-fittedV_err), Vpeak*(fittedV+fittedV_err),
                     facecolor=color_dict[band], alpha=0.1)
#plt.xlim(min(min(t_extrapV)-3., -25.+mjd_Vpeak),10.+mjd_Vpeak)
plt.xlim(57982, 57990)

plt.legend()
plt.yscale('log')
plt.xlim(57982, 57990)
plt.show()
plt.close(fig)
    
# original_phot.to_csv(OUTPUT_PATH+'/%s.dat'%snname, na_rep='nan',
#               index=False, sep='\t')

# --- CONVERT DATA TO LOG-SPACE BEFORE SAVING ---
# 1. Calculate Phase (time since explosion)
t0_fix = explosion_dates[snname][0]
original_phot['Phase'] = original_phot['MJD'] - t0_fix

# 2. Filter out pre-explosion points (Phase < 0) which break log-time
original_phot = original_phot[original_phot['Phase'] >= 0].copy()

# 3. Safeguard against log10(0) = -inf at the exact explosion time
original_phot.loc[original_phot['Phase'] == 0, 'Phase'] = 1e-5 

# 4. Safeguard against log10(0) = -inf for the zero-flux points we just modeled
original_phot = original_phot[original_phot['Flux'] >= 0].copy()
original_phot.loc[original_phot['Flux'] == 0, 'Flux'] = 1e-25

# 5. Apply transformations
original_phot['Log_Phase'] = np.log10(original_phot['Phase'])
original_phot['Log_Flux'] = np.log10(original_phot['Flux'])

# 6. Propagate flux errors into log space: error_log = error_linear / (Flux * ln(10))
original_phot['Log_Flux_err'] = original_phot['Flux_err'] / (original_phot['Flux'] * np.log(10))

# --- FINAL SANITIZATION BEFORE SAVING ---
# 1. Catch infs or NaNs in the linear flux error and fallback to 10% of the flux
original_phot['Flux_err'] = np.where(
    np.isinf(original_phot['Flux_err']) | np.isnan(original_phot['Flux_err']), 
    0.1 * np.abs(original_phot['Flux']), 
    original_phot['Flux_err']
)

# 2. Recalculate the log flux error using the newly sanitized linear error
if 'Log_Flux' in original_phot.columns:
    original_phot['Log_Flux_err'] = original_phot['Flux_err'] / (original_phot['Flux'] * np.log(10))

# 3. Absolute catch-all: replace any straggling infs with NaN so they don't break downstream tools
original_phot.replace([np.inf, -np.inf], np.nan, inplace=True)

# Save the final log-ready dataframe
original_phot.to_csv(OUTPUT_PATH+'/%s.dat'%snname, na_rep='nan',
              index=False, sep='\t')

# Save the final log-ready dataframe
original_phot.to_csv(OUTPUT_PATH+'/%s.dat'%snname, na_rep='nan',
              index=False, sep='\t')