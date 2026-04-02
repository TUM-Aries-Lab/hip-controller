%% =========================================================
%  Batch Simulink Runner
%  - Iterates over all CSV files in a folder
%  - Sets model inputs and stop time from each CSV
%  - Runs simulation and exports results
%% =========================================================
scriptFolder = fileparts(mfilename('fullpath'));
% --- CONFIGURATION (edit these) ---
csvFolder    = fullfile(scriptFolder, 'input_data/');  % folder containing input CSVs
outputFolder = fullfile(scriptFolder, 'output_data/'); % folder to save results
modelName    = 'final_version_control_2026_03_26';            % Simulink model name (no .slx)
signalNames  = {'motor_command_left', 'motor_command_right', 'vel_L', 'vel_R', 'angle_L', 'angle_R', 'gait_phase_left', 'gait_phase_right', 'amplitude_left', 'amplitude_right'};     % signals to export from Data Inspector
% ----------------------------------

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Load the Simulink model (load once, run many)
load_system(modelName);

% Get all CSV files recursively through all subdirectories
files = dir(fullfile(csvFolder, '**', '*.csv'));

if isempty(files)
    error('No CSV files found in: %s', csvFolder);
end

fprintf('Found %d CSV file(s). Starting batch run...\n\n', length(files));

%% --- Main Loop ---
for i = 1:length(files)

    fileName = files(i).name;
    [~, baseName, ~] = fileparts(fileName);
    filePath = fullfile(files(i).folder, fileName);

    % Reconstruct relative subdirectory path and mirror it in output folder
    relativeDir = strrep(files(i).folder, csvFolder, '');
    thisOutputFolder = fullfile(outputFolder, relativeDir);

    % Create the mirrored subdirectory if it doesn't exist
    if ~exist(thisOutputFolder, 'dir')
        mkdir(thisOutputFolder);
    end

    fprintf('[%d/%d] Processing: %s\n', i, length(files), fileName);

    % -------------------------------------------------
    % 1. Import CSV
    % -------------------------------------------------
    try
        T = readtable(filePath, 'VariableNamingRule', 'preserve');
    catch ME
        warning('Failed to read %s: %s. Skipping.\n', fileName, ME.message);
        continue;
    end

    % -------------------------------------------------
    % 2. Compute workspace values from table
    % -------------------------------------------------
    angle_left  = T.("angle_left (rad)");
    angle_right = T.("angle_right (rad)");
    angles      = [angle_left, angle_right];
    angles      = single(angles);

    input                         = Simulink.Parameter;
    input.Value                   = angles;
    input.CoderInfo.StorageClass  = 'Auto';

    simDuration = (input.Dimensions(1) - 1) / 100;

    % -------------------------------------------------
    % 3. Push values into the Simulink model
    % -------------------------------------------------
    % 'input' is assigned to the base workspace so the model can see it
    assignin('base', 'input', input);

    % Set the simulation stop time
    set_param(modelName, 'StopTime', num2str(simDuration));

    % -------------------------------------------------
    % 4. Run the simulation
    % -------------------------------------------------
    try
        fprintf('   Running simulation (stop time: %ds)...\n', simDuration);
        simOut = sim(modelName);
    catch ME
        warning('Simulation failed for %s: %s. Skipping.\n', fileName, ME.message);
        continue;
    end

    % -------------------------------------------------
    % 5. Export results
    % -------------------------------------------------

    % Extract all signals and save as a single CSV
    try
        timeVec  = simOut.logsout.getElement(signalNames{1}).Values.Time;

        % Start table with time and input angles (already in workspace)
        outTable = table(timeVec, angles(:,1), angles(:,2), ...
                        'VariableNames', {'Time', 'raw_angle_left_rad', 'raw_angle_right_rad'});

        % Append remaining signals from logsout
        for s = 1:length(signalNames)
            sigName = signalNames{s};
            try
                sigData = simOut.logsout.getElement(sigName).Values;
                outTable.(sigName) = sigData.Data;
            catch
                warning('   Signal "%s" not found in logsout. Skipping.\n', sigName);
            end
        end

        csvOutPath = fullfile(thisOutputFolder, [baseName '.csv']);
        writetable(outTable, csvOutPath);
        fprintf('   Saved combined signals -> %s\n', csvOutPath);

    catch ME
        warning('Export failed for %s: %s\n', fileName, ME.message);
    end

    fprintf('   Done.\n\n');
end

%% --- Cleanup ---
close_system(modelName, 0);   % 0 = don't save changes to the model
fprintf('Batch run complete. Results saved to: %s\n', outputFolder);
