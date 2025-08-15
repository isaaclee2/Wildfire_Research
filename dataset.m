% Generate training data for problem of predicting fire area based on ignition time weather conditions, static data, and forecast length
% Use WRF-SFIRE solutions for 2023 wildfires, along with corresponding weather data from met_em files, terrain data, and fuel category data

% Input scalar values: forecast time length, avg. U-wind, avg. V-wind, avg. relative humidity, avg. temperature, avg. terrain gradient in x-dir., avg. terrain gradient in y-dir., max terrain height variation, terrain RMS (roughness), num. of fuel categories
% Output scalar value: fire area after time T

% Data array with be size N x (length(x) + length(y)); here N = 152*10*10 samples, length(x) = 1, and length(y) = 23
% For averaging variables, average across center 800 x 800 pixels that will always have non-zero values after data augmentation rotations (i.e., 30/2 km * sqrt(2) is max domain side length from center of WRF-SFIRE simulation domain that will have non-zero values after rotations, so choose 20 x 20 km box from center of WRF-SFIRE domain to consider)
% For finding weather conditions at ignition time, linearly interpolate between avaiable met_em times to find conditions at the ignition time prescibed for each WRF-SFIRE simulation

% Output fire area will be reported in acres; normalize by dividing by ...

% To select forecast length, choose from U(0,max(arr_time)) per sample; normalize by dividing by 48 h
% For weather data use NAM grid 227 data (5 km resolution data at 1 hour intervals); for training have NAM227 data at 3h intervals interpolated to 1km resolution from met_em files corresponding to WRF-SFIRE solutions (lowest level of UU corresponds to 10m winds); make sure to project back to proper u and v coordinates after doing rotations; normalize by adding 12 and dividing by 24; 
% For relative humidity use same NAM 227 data as used for wind (lowest level of RH in met_em files corresponds to 2 m relative humidity); process the same as wind data; normalize by dividing by 110
% For temperature use same NAM 227 data as used for wind (lowest level of TT in met_em files corresponds to 2 m temperature in K); process the same as wind data; normalize by dividing by ...
% Average terrain gradients are found by taking gradients in x and y directions separately; make sure to project back onto proper coordinates after doing rotations; terrain gradients are normalized by ...
% Maximum terrain variation is found by subtracting minimum terrain height from maximum terrain height for domain; max terrain variation is normalized by .. (dividing by 1200 m ?)
% Fuel categories (categorical values from 1-14); for each category consider total number of pixels with that fuel category; when doing rotations for data augmentation threshold pixel values at 0.5; normalize by dividing by 640000 (i.e., 800*800 which is total number of pixels)


%% generate training data
clear; close all; clc;

wrfout_dir = "/Users/bshaddy/CD3_lab_stuff/wildfire_research/wrf-sfire_training_data_w_terrain/simulation_result_files";
wrfout_files = dir(wrfout_dir+ "/*.nc");
met_em_dir = "/Users/bshaddy/CD3_lab_stuff/wildfire_research/wrf-sfire_training_data_w_terrain/atm_data_for_simulation_result_files/met_em_files_for_WRF-SFIRE_simulations";

% rotation augments; forecast lengths T per augment;
augments = 10;
forecast_lengths_per_augment = 10;

data_vec = zeros(length(wrfout_files(:))*augments*forecast_lengths_per_augment,24);    % N x (length(x) + length(y))

sample_counter = 1;
for Case = 1:length(wrfout_files(:))
    % print case number to track progress
    disp("Case: " +Case);

    % simulation start date and time
    simulation_start_date = ncreadatt(wrfout_dir+ "/" +wrfout_files(Case).name,"/","SIMULATION_START_DATE");
    simulation_start_day = str2double(simulation_start_date(9:10));
    simulation_start_time = str2double(simulation_start_date(12:13));

    % fire arrival times
    tign_g = ncread(wrfout_dir+ "/" +wrfout_files(Case).name,'TIGN_G');
    tign_g = tign_g(1:1200,1:1200)/3600;
    tign_g = tign_g + simulation_start_time;                               % make arrival times relative to 0 UTC on ignition day
    tign_g = rot90(tign_g);                                                % rotate to align E-W & N-S
    ign_time = min(tign_g,[],'all');                                       % get ignition time for determining met_em files to use 

    % find correct atmospheric data files and their respective times
    met_em_files = dir(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/*.nc");
    met_em_ign_time_files = met_em_files(1:2);
    for k = 1:length(met_em_ign_time_files)
        met_em_days(k) = str2double(met_em_ign_time_files(k).name(20:21));
        met_em_times(k) = str2double(met_em_ign_time_files(k).name(23:24));
        if met_em_days(k) ~= simulation_start_day
            met_em_times(k) = met_em_times(k) + 24;
        end
    end
    kk = 2;
    while ign_time > met_em_times(2)
        disp(kk);
        met_em_ign_time_files = met_em_files(kk:kk+1);
        for k = 1:length(met_em_ign_time_files)
            met_em_days(k) = str2double(met_em_ign_time_files(k).name(20:21));
            met_em_times(k) = str2double(met_em_ign_time_files(k).name(23:24));
            if met_em_days(k) ~= simulation_start_day
                met_em_times(k) = met_em_times(k) + 24;
            end
        end
        kk = kk+1;
    end

    % find linear interpolation weightings for met_em files to get values at ignition time
    met_em_file_2_wgt = (ign_time - met_em_times(1)) / 3;
    met_em_file_1_wgt = 1 - met_em_file_2_wgt;

    % make sure have proper met_em files via weightings
    if met_em_file_1_wgt>1 || met_em_file_1_wgt<0 || met_em_file_2_wgt>1 || met_em_file_2_wgt<0
        disp('wrong met_em files');
        return
    end

    % Load UU, VV, RH, TT from ignition time met_em files
    UU1 = ncread(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/" +met_em_ign_time_files(1).name,"UU");
    VV1 = ncread(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/" +met_em_ign_time_files(1).name,"VV");
    RH1 = ncread(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/" +met_em_ign_time_files(1).name,"RH");
    TT1 = ncread(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/" +met_em_ign_time_files(1).name,"TT");
    UU2 = ncread(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/" +met_em_ign_time_files(2).name,"UU");
    VV2 = ncread(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/" +met_em_ign_time_files(2).name,"VV");
    RH2 = ncread(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/" +met_em_ign_time_files(2).name,"RH");
    TT2 = ncread(met_em_dir+ "/" +wrfout_files(Case).name(1:end-3)+ "_met_em/" +met_em_ign_time_files(2).name,"TT");

    % linearly interpolate UU, VV, RH, TT 
    UU = UU1*met_em_file_1_wgt + UU2*met_em_file_2_wgt;
    VV = VV1*met_em_file_1_wgt + VV2*met_em_file_2_wgt;
    RH = RH1*met_em_file_1_wgt + RH2*met_em_file_2_wgt;
    TT = TT1*met_em_file_1_wgt + TT2*met_em_file_2_wgt;

    % set any negative RH values to 0 since this is not possible
    if min(RH,[],'all') < 0
        RH(RH<0) = 0;                  
        disp("RH negative");
    end

    % Resample UU, VV, RH and take lowest vertical level to get 10m u, 10m v, 2m rh, 2m t
    u10 = imresize(UU(1:30,1:30,1),1000/25,'box');
    v10 = imresize(VV(1:30,1:30,1),1000/25,'box');   
    rh = imresize(RH(1:30,1:30,1),1000/25,'box');
    t = imresize(TT(1:30,1:30,1),1000/25,'box');
    u10 = rot90(u10);                                          % rotate to align E-W & N-S
    v10 = rot90(v10);                                          % rotate to align E-W & N-S
    rh = rot90(rh);                                            % rotate to align E-W & N-S
    t = rot90(t);                                              % rotate to align E-W & N-S

    % terrain height data
    zsf = ncread(wrfout_dir+ "/" +wrfout_files(Case).name,'ZSF');
    zsf = zsf(1:1200,1:1200);
    zsf = rot90(zsf);                                                      % rotate to align E-W & N-S

    % fuel category data
    nfuel_cat = ncread(wrfout_dir+ "/" +wrfout_files(Case).name,'NFUEL_CAT');
    nfuel_cat = nfuel_cat(1:1200,1:1200);
    nfuel_cat = rot90(nfuel_cat);                                          % rotate to align E-W & N-S

    % fuel category binary masks
    nfuel_cat_binary_masks = create_nfuel_cat_binary_masks(nfuel_cat);

    for augment = 1:augments
        % augmentation; rotates randomly between 0 and 360 deg
        rotation = 360*rand;
        rotated_tign_g = imrotate(tign_g,rotation,'nearest','loose');      
        rotated_u10 = imrotate(u10,rotation,'nearest','loose'); 
        rotated_v10 = imrotate(v10,rotation,'nearest','loose'); 
        rotated_rh = imrotate(rh,rotation,'nearest','loose'); 
        rotated_t = imrotate(t,rotation,'nearest','loose'); 
        rotated_zsf = imrotate(zsf,rotation,'nearest','loose'); 
        rotated_nfuel_cat_binary_masks = imrotate(nfuel_cat_binary_masks,rotation,'nearest','loose'); 

        % Project wind back to u and v coordinates; project terrain gradients
        rotated_proj_u10 = rotated_u10*cosd(rotation) - rotated_v10*cosd(90-rotation);
        rotated_proj_v10 = rotated_u10*sind(rotation) + rotated_v10*sind(90-rotation);

        % clip binary fuel mask values
        rotated_nfuel_cat_binary_masks(rotated_nfuel_cat_binary_masks >= 0.5) = 1;
        rotated_nfuel_cat_binary_masks(rotated_nfuel_cat_binary_masks < 0.5) = 0;        

        % crop variables to center 800 x 800 pixels of domain
        crop_row_col_start = floor(length(rotated_tign_g)/2) - 399;
        crop_row_col_end = crop_row_col_start + 799;
        cropped_rotated_tign_g = rotated_tign_g(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
        cropped_rotated_proj_u10 = rotated_proj_u10(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
        cropped_rotated_proj_v10 = rotated_proj_v10(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
        cropped_rotated_rh = rotated_rh(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
        cropped_rotated_t = rotated_t(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
        cropped_rotated_zsf = rotated_zsf(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end);
        cropped_rotated_nfuel_cat_binary_masks = rotated_nfuel_cat_binary_masks(crop_row_col_start:crop_row_col_end,crop_row_col_start:crop_row_col_end,:);
        
        % take averages
        avg_u10 = mean(cropped_rotated_proj_u10,'all');
        avg_v10 = mean(cropped_rotated_proj_v10,'all');
        avg_rh = mean(cropped_rotated_rh,'all');
        avg_t = mean(cropped_rotated_t,'all');

        % compute terrain metrics (avg. gradients, max variations, RMS)
        [terr_grad_x,terr_grad_y] = gradient(cropped_rotated_zsf,25);                                          % terrain gradients as [m/m]; uses 25m grid spacing
        avg_terr_grad_x = mean(terr_grad_x,'all');                                                             % average terrain gradient in x-direction 
        avg_terr_grad_y = mean(terr_grad_y,'all');                                                             % average terrain gradient in y-direction
        max_terr_var = max(cropped_rotated_zsf,[],'all') - min(cropped_rotated_zsf,[],'all');                  % maximum difference in terrain height (i.e., max hgt - min hgt)
        terr_rms_roughness = rms(cropped_rotated_zsf-mean(cropped_rotated_zsf,'all'),'all');                   % RMS roughness value of terrain height relative to mean terrain height

        % count numbers of fuel categories
        num_of_fuel_types = squeeze(sum(cropped_rotated_nfuel_cat_binary_masks,[1,2]));

        for k = 1:forecast_lengths_per_augment
            % select random forecast length
            forecast_length_tign_g = cropped_rotated_tign_g - min(cropped_rotated_tign_g,[],'all');
            forecast_length = max(forecast_length_tign_g,[],'all') * rand;