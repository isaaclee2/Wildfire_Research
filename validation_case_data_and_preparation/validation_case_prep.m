% Generate training data for problem of recursively predicting progression of scalar fire area over initial 24 hours of a fire based on current fire size, fire size at previous forecast step, weather conditions, and static data
% Use WRF-SFIRE solutions for 2023 wildfires, along with corresponding weather data from met_em files, terrain data, and fuel category data

% Input scalar values: fire area at current forecast step, fire area at previous forecast step, avg. U-wind, avg. V-wind, avg. relative humidity, avg. temperature, avg. terrain gradient in x-dir., avg. terrain gradient in y-dir., max terrain height variation, terrain RMS (roughness), num. of fuel categories
% Output scalar fire area value vector: change in fire area after 1, 2, and 3 hours from current forecast step

% Data array with be size N x (length(x) + length(y)); here N = 152*10*10 samples, length(x) = 3, and length(y) = 24
% For averaging variables, average across center 300 x 300 pixels (7.5 km x 7.5 km) that will always have non-zero values after data augmentation rotations (i.e., 30/2 km * sqrt(2) is max domain side length from center of WRF-SFIRE simulation domain that will have non-zero values after rotations)
% For finding weather conditions at forecast step time, linearly interpolate between avaiable met_em times to find conditions at the prescibed time 

% Output fire area change values will be reported in acres; normalize by dividing by ...

% To select forecast times, choose from U(ign_time,ign_time+24-3) per sample
% For weather data use NAM grid 227 data (5 km resolution data at 1 hour intervals); for training have NAM227 data at 3h intervals interpolated to 1km resolution from met_em files corresponding to WRF-SFIRE solutions (lowest level of UU corresponds to 10m winds); make sure to project back to proper u and v coordinates after doing rotations; normalize by adding 13.5 and dividing by 27; 
% For relative humidity use same NAM 227 data as used for wind (lowest level of RH in met_em files corresponds to 2 m relative humidity); process the same as wind data; normalize by dividing by 100
% For temperature use same NAM 227 data as used for wind (lowest level of TT in met_em files corresponds to 2 m temperature in K); process the same as wind data; normalize by dividing by subtracting 260 and dividing by 60
% Average terrain gradients are found by taking gradients in x and y directions separately; make sure to project back onto proper coordinates after doing rotations; terrain gradients are normalized by adding 0.14 and dividing by 0.28
% Maximum terrain variation is found by subtracting minimum terrain height from maximum terrain height for domain; max terrain variation is normalized by dividing by 2500 m
% Terrain RMS is computed and normalized by dividing by 425
% Fuel categories (categorical values from 1-14); for each category consider total number of pixels with that fuel category; when doing rotations for data augmentation threshold pixel values at 0.5; normalize by dividing by 90000 (i.e., 300*300 which is total number of pixels)
clear; close all; clc;

% wrfout_file = '/Users/bshaddy/CD3_lab_stuff/wildfire_research/wrf-sfire_training_data_w_terrain/predict_fire_area_from_ign_time_conditions_(Isaac)/validation_case_data_and_preparation/wrfout_files/Horse/wrfout_d01_2025-07-04_09:00:00';
wrfout_file = '/Users/isaaclee/Wildfire_Research/validation_case_data_and_preparation/wrfout_files/Horse/wrfout_d01_2025-07-04_09:00:00';
% met_em_dir = '/Users/bshaddy/CD3_lab_stuff/wildfire_research/wrf-sfire_training_data_w_terrain/predict_fire_area_from_ign_time_conditions_(Isaac)/validation_case_data_and_preparation/met_em_data/Horse';
met_em_dir = '/Users/isaaclee/Wildfire_Research/validation_case_data_and_preparation/met_em_data/Horse';

% sample params 
dT = 3;                % prediction timestep in hours
max_forecast = 36;     % maximum forecast length

data_vec = zeros(max_forecast/dT-1,27);    % N x (length(x) + length(y))

sample_counter = 1;

% simulation start date and time
simulation_start_date = ncreadatt(wrfout_file,"/","SIMULATION_START_DATE");
simulation_start_month = str2double(simulation_start_date(6:7));
simulation_start_day = str2double(simulation_start_date(9:10));
simulation_start_time = str2double(simulation_start_date(12:13));

% fire arrival times
tign_g = double(ncread(wrfout_file,'TIGN_G'));
tign_g = tign_g(1:1200,1:1200)/3600;
tign_g = tign_g + simulation_start_time;                               % make arrival times relative to 0 UTC on simulation start day
tign_g = rot90(tign_g);                                                % rotate to align E-W & N-S

% terrain height data
zsf = double(ncread(wrfout_file,'ZSF'));
zsf = zsf(1:1200,1:1200);
zsf = rot90(zsf);                                                      % rotate to align E-W & N-S

% fuel category data
nfuel_cat = double(ncread(wrfout_file,'NFUEL_CAT'));
nfuel_cat = nfuel_cat(1:1200,1:1200);
nfuel_cat = rot90(nfuel_cat);                                          % rotate to align E-W & N-S

% fuel category binary masks
nfuel_cat_binary_masks = create_nfuel_cat_binary_masks(nfuel_cat);

ign_time = min(tign_g,[],'all');             % get ignition time relative to 0 UTC on simulation start day, for determining met_em files to use 

forecast_times = zeros(1,max_forecast/dT-1);
for k = 1:max_forecast/dT-1
    forecast_times(k) = ign_time + (k-1)*dT;
end

for forecast_T = forecast_times

    % determine fire area for previous forecast step, current forecast step, and hourly predictions over next forecast step
    [idx_prev] = find(tign_g <= (forecast_T-dT));
    fire_area_prev = length(idx_prev) * 25^2;                                % fire area in m^2
    fire_area_prev = fire_area_prev / 4046.856422;                           % fire area in acres
    [idx_curr] = find(tign_g <= (forecast_T));
    fire_area_curr = length(idx_curr) * 25^2;                                % fire area in m^2
    fire_area_curr = fire_area_curr / 4046.856422;                           % fire area in acres
    [idx_pred_1] = find(tign_g <= (forecast_T+dT*1/3));
    fire_area_pred_1 = length(idx_pred_1) * 25^2;                            % fire area in m^2
    fire_area_pred_1 = fire_area_pred_1 / 4046.856422;                       % fire area in acres
    [idx_pred_2] = find(tign_g <= (forecast_T+dT*2/3));
    fire_area_pred_2 = length(idx_pred_2) * 25^2;                            % fire area in m^2
    fire_area_pred_2 = fire_area_pred_2 / 4046.856422;                       % fire area in acres
    [idx_pred_3] = find(tign_g <= (forecast_T+dT*3/3));
    fire_area_pred_3 = length(idx_pred_3) * 25^2;                            % fire area in m^2
    fire_area_pred_3 = fire_area_pred_3 / 4046.856422;                       % fire area in acres

    % Get fire area prediction as difference from current area after 1, 2, 3 hours
    fire_area_pred_change_1 = fire_area_pred_1 - fire_area_curr;
    fire_area_pred_change_2 = fire_area_pred_2 - fire_area_curr;
    fire_area_pred_change_3 = fire_area_pred_3 - fire_area_curr;

    % find atmospheric data file indices and their weights for current forecast time 
    met_em_files = dir(met_em_dir+ "/*.nc");
    met_em_idx_start = 1;
    met_em_forecast_time_files = met_em_files(met_em_idx_start:met_em_idx_start+1);
    for k = 1:length(met_em_forecast_time_files)
        met_em_months(k) = str2double(met_em_forecast_time_files(k).name(17:18));
        met_em_days(k) = str2double(met_em_forecast_time_files(k).name(20:21));
        met_em_times(k) = str2double(met_em_forecast_time_files(k).name(23:24));
        if met_em_days(k) ~= simulation_start_day
            if met_em_months(k) == simulation_start_month
                day_diff = met_em_days(k) - simulation_start_day;
            else
                met_em_day_diff_days = [];
                for days = 1:length(met_em_files)
                    met_em_day_diff_days = cat(1,met_em_day_diff_days,str2double(met_em_files(days).name(20:21)));
                end
                met_em_day_diff_days_max = max(met_em_day_diff_days);
                day_diff = met_em_day_diff_days_max + met_em_days(k) - simulation_start_day;
            end
            met_em_times(k) = met_em_times(k) + 24*day_diff;
        end
    end
    while (forecast_T+dT/2) > met_em_times(2)
        met_em_idx_start = met_em_idx_start + 1;
        met_em_forecast_time_files = met_em_files(met_em_idx_start:met_em_idx_start+1);
        for k = 1:length(met_em_forecast_time_files)
            met_em_months(k) = str2double(met_em_forecast_time_files(k).name(17:18));
            met_em_days(k) = str2double(met_em_forecast_time_files(k).name(20:21));
            met_em_times(k) = str2double(met_em_forecast_time_files(k).name(23:24));
            if met_em_days(k) ~= simulation_start_day
                if met_em_months(k) == simulation_start_month
                    day_diff = met_em_days(k) - simulation_start_day;
                else
                    met_em_day_diff_days = [];
                    for days = 1:length(met_em_files)
                        met_em_day_diff_days = cat(1,met_em_day_diff_days,str2double(met_em_files(days).name(20:21)));
                    end
                    met_em_day_diff_days_max = max(met_em_day_diff_days);
                    day_diff = met_em_day_diff_days_max + met_em_days(k) - simulation_start_day;
                end
                met_em_times(k) = met_em_times(k) + 24*day_diff;
            end
        end
    end

    % find linear interpolation weightings for met_em files to get values at half way point between forecasting steps
    met_em_file_2_wgt = ((forecast_T+dT/2) - met_em_times(1)) / dT;
    met_em_file_1_wgt = 1 - met_em_file_2_wgt;

    % make sure have proper met_em files via weightings
    if met_em_file_1_wgt>1 || met_em_file_1_wgt<0 || met_em_file_2_wgt>1 || met_em_file_2_wgt<0
        disp('wrong met_em files');
        return
    end

    % met_em files
    met_em_file_1 = met_em_files(met_em_idx_start).name;
    met_em_file_2 = met_em_files(met_em_idx_start+1).name;

    % Load UU, VV, RH for the two relevant met_em times
    UU1 = double(ncread(met_em_dir+ "/" +met_em_file_1,"UU"));
    VV1 = double(ncread(met_em_dir+ "/" +met_em_file_1,"VV"));
    RH1 = double(ncread(met_em_dir+ "/" +met_em_file_1,"RH"));
    TT1 = double(ncread(met_em_dir+ "/" +met_em_file_1,"TT"));
    UU2 = double(ncread(met_em_dir+ "/" +met_em_file_2,"UU"));
    VV2 = double(ncread(met_em_dir+ "/" +met_em_file_2,"VV"));
    RH2 = double(ncread(met_em_dir+ "/" +met_em_file_2,"RH"));
    TT2 = double(ncread(met_em_dir+ "/" +met_em_file_2,"TT"));

    % set any negative RH values to 0 since this is not possible
    if min(RH1,[],'all') < 0
        RH1(RH1<0) = 0;                  
        % disp("RH negative");
    end
    if min(RH2,[],'all') < 0
        RH2(RH2<0) = 0;                  
        % disp("RH negative");
    end

    % linearly interpolate UU, VV, RH, TT 
    UU = UU1*met_em_file_1_wgt + UU2*met_em_file_2_wgt;
    VV = VV1*met_em_file_1_wgt + VV2*met_em_file_2_wgt;
    RH = RH1*met_em_file_1_wgt + RH2*met_em_file_2_wgt;
    TT = TT1*met_em_file_1_wgt + TT2*met_em_file_2_wgt;

    % Resample UU, VV, RH and take lowest vertical level to get 10m u, 10m v, 2m rh, 2m t
    u10 = imresize(UU(1:30,1:30,1),1000/25,'box');
    v10 = imresize(VV(1:30,1:30,1),1000/25,'box');   
    rh = imresize(RH(1:30,1:30,1),1000/25,'box');
    t = imresize(TT(1:30,1:30,1),1000/25,'box');
    u10 = rot90(u10);                                          % rotate to align E-W & N-S
    v10 = rot90(v10);                                          % rotate to align E-W & N-S
    rh = rot90(rh);                                            % rotate to align E-W & N-S
    t = rot90(t);                                              % rotate to align E-W & N-S   

    % crop variables to center 300 x 300 pixels of domain (7.5 km x 7.5 km)
    crop_row_col_start = floor(length(tign_g)/2) - 149;
    crop_row_col_end = crop_row_col_start + 299;
    cropped_u10 = u10(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
    cropped_v10 = v10(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
    cropped_rh = rh(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
    cropped_t = t(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
    cropped_zsf = zsf(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
    cropped_nfuel_cat_binary_masks = nfuel_cat_binary_masks(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end,:);
    
    % take averages
    avg_u10 = mean(cropped_u10,'all');
    avg_v10 = mean(cropped_v10,'all');
    avg_rh = mean(cropped_rh,'all');
    avg_t = mean(cropped_t,'all');

    % compute terrain metrics (avg. gradients, max variations, RMS)
    [terr_grad_x,terr_grad_y] = gradient(cropped_zsf,25);                                          % terrain gradients as [m/m]; uses 25m grid spacing
    avg_terr_grad_x = mean(terr_grad_x,'all');                                                             % average terrain gradient in x-direction 
    avg_terr_grad_y = mean(terr_grad_y,'all');                                                             % average terrain gradient in y-direction
    max_terr_var = max(cropped_zsf,[],'all') - min(cropped_zsf,[],'all');                  % maximum difference in terrain height (i.e., max hgt - min hgt)
    terr_rms_roughness = rms(cropped_zsf-mean(cropped_zsf,'all'),'all');                   % RMS roughness value of terrain height relative to mean terrain height  

    % count numbers of fuel categories
    num_of_fuel_types = squeeze(sum(cropped_nfuel_cat_binary_masks,[1,2]));

    % normalize
    data_vec(sample_counter,1) = fire_area_pred_change_1;          % normalize by ... 
    data_vec(sample_counter,2) = fire_area_pred_change_2;  
    data_vec(sample_counter,3) = fire_area_pred_change_3;  

    data_vec(sample_counter,4) = 0;             % normalize by ...
    data_vec(sample_counter,5) = 0;

    data_vec(sample_counter,6) = (avg_u10+13.5)/27;                % normalize using approximate max and min wind speeds
    data_vec(sample_counter,7) = (avg_v10+13.5)/27;                % normalize using approximate max and min wind speeds
    data_vec(sample_counter,8) = avg_rh/100;                       % normalize by approximate max relative humidity of 110
    data_vec(sample_counter,9) = (avg_t-260)/60;                   % normalize by subtracting 260 and dividing by 60       
    data_vec(sample_counter,10) = (avg_terr_grad_x+0.14)/0.28;     % normalize by adding 0.06 and dividing by 0.12
    data_vec(sample_counter,11) = (avg_terr_grad_y+0.14)/0.28;     % normalize by adding 0.06 and dividing by 0.12
    data_vec(sample_counter,12) = max_terr_var/2500;               % normalizy by dividing by 3200                      
    data_vec(sample_counter,13) = terr_rms_roughness/425;          % normalize by dividing by 510
    data_vec(sample_counter,14:27) = num_of_fuel_types/(300*300);  % normalize by total number of pixels in 300x300 portion of domain considered 
         
    sample_counter = sample_counter+1;
end 

% save
save("/Users/isaaclee/Wildfire_Research/validation_case_data_and_preparation/validation_case_data.mat",'data_vec');
% import py.numpy
% py.numpy.save("/Users/bshaddy/CD3_lab_stuff/wildfire_research/wrf-sfire_training_data_w_terrain/predict_fire_area_from_ign_time_conditions_(Isaac)/validation_case_data_and_preparation/validation_case_data.npy",data_vec);


%% Functions 
function nfuel_cat_binary_masks = create_nfuel_cat_binary_masks(nfuel_cat)
    nfuel_cat_1 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_2 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_3 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_4 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_5 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_6 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_7 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_8 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_9 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_10 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_11 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_12 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_13 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_14 = zeros(size(nfuel_cat),'like',nfuel_cat);
    nfuel_cat_1(nfuel_cat == 1) = 1;
    nfuel_cat_1(nfuel_cat ~= 1) = 0;
    nfuel_cat_2(nfuel_cat == 2) = 1;
    nfuel_cat_2(nfuel_cat ~= 2) = 0;
    nfuel_cat_3(nfuel_cat == 3) = 1;
    nfuel_cat_3(nfuel_cat ~= 3) = 0;
    nfuel_cat_4(nfuel_cat == 4) = 1;
    nfuel_cat_4(nfuel_cat ~= 4) = 0;
    nfuel_cat_5(nfuel_cat == 5) = 1;
    nfuel_cat_5(nfuel_cat ~= 5) = 0;
    nfuel_cat_6(nfuel_cat == 6) = 1;
    nfuel_cat_6(nfuel_cat ~= 6) = 0;
    nfuel_cat_7(nfuel_cat == 7) = 1;
    nfuel_cat_7(nfuel_cat ~= 7) = 0;
    nfuel_cat_8(nfuel_cat == 8) = 1;
    nfuel_cat_8(nfuel_cat ~= 8) = 0;
    nfuel_cat_9(nfuel_cat == 9) = 1;
    nfuel_cat_9(nfuel_cat ~= 9) = 0;
    nfuel_cat_10(nfuel_cat == 10) = 1;
    nfuel_cat_10(nfuel_cat ~= 10) = 0;
    nfuel_cat_11(nfuel_cat == 11) = 1;
    nfuel_cat_11(nfuel_cat ~= 11) = 0;
    nfuel_cat_12(nfuel_cat == 12) = 1;
    nfuel_cat_12(nfuel_cat ~= 12) = 0;
    nfuel_cat_13(nfuel_cat == 13) = 1;
    nfuel_cat_13(nfuel_cat ~= 13) = 0;
    nfuel_cat_14(nfuel_cat == 14) = 1;
    nfuel_cat_14(nfuel_cat ~= 14) = 0;
    nfuel_cat_binary_masks = cat(3,nfuel_cat_1,nfuel_cat_2,nfuel_cat_3,nfuel_cat_4,nfuel_cat_5,nfuel_cat_6,nfuel_cat_7,nfuel_cat_8,nfuel_cat_9,nfuel_cat_10,nfuel_cat_11,nfuel_cat_12,nfuel_cat_13,nfuel_cat_14);
end
