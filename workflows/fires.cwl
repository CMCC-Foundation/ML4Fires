#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: Workflow
label: Fires

requirements:
  MultipleInputFeatureRequirement: {}
  InlineJavascriptRequirement: {}
  StepInputExpressionRequirement: {}

inputs:
  inputexperiment:
    type: File?
  nthreads: int
  container: string
  time_range: string

outputs:
  outputexperiment:
    type: File
    outputSource: End_iteration_on_scenarios/experiment

steps:
  Init_frequency:
    run: tasks/set.cwl
    in:
      experiment: inputexperiment
      name:
        default: "Init frequency"
      key:
        default: "frequency"
      value:
        default: "Eday|day|day|day|day|fx"
    out: [experiment]
  Init_measure:
    run: tasks/set.cwl
    in:
      experiment: Init_frequency/experiment
      name:
        default: "Init measure"
      key:
        default: "measure"
      value:
        default: "lai|lst_day|rel_hum|t2m_min|pr|lsm"
    out: [experiment]
  Create_a_work_container:
    run: tasks/createcontainer.cwl
    in:
      experiment: Init_measure/experiment
      name:
        default: "Create a work container"
      container: container
      dim:
        default: "time|plev|lat|lon"
      hierarchy:
        default: "oph_time|oph_base|oph_base|oph_base"
      on_error:
        default: "skip"
    out: [experiment]

  Import_mask:
    run: tasks/importnc2.cwl
    in:
      experiment: Create_a_work_container/experiment
      name:
        default: "Import mask"
      imp_dim:
        default: "time"
      measure:
        default: "basis_regions"
      src_path:
        default: "/home/jovyan/work/fires/mask.nc"
      container: container
      nfrag: nthreads
      nthreads: nthreads
    out: [experiment]

  Iterate_on_scenarios:
    run: tasks/for.cwl
    in:
      experiment: Create_a_work_container/experiment
      name:
        default: "Iterate on scenarios"
      parallel:
        default: "yes"
      key:
        default: "scenario"
      values:
        default: "ssp126"
    out: [experiment]

  Iterate_on_models:
    run: tasks/for.cwl
    in:
      experiment: Iterate_on_scenarios/experiment
      name:
        default: "Iterate on models"
      parallel:
        default: "yes"
      key:
        default: "model"
      values:
        default: "CMCC-ESM2"
    out: [experiment]

  Iterate_on_variables:
    run: tasks/for.cwl
    in:
      experiment: Iterate_on_models/experiment
      name:
        default: "Iterate on variables"
      parallel:
        default: "yes"
      key:
        default: "variable"
      values:
        default: "lai|tas|hur|tasmin|pr|sftlf"
    out: [experiment]

  Check_for_reduction_operation:
    run: tasks/if.cwl
    in:
      experiment: Iterate_on_variables/experiment
      name:
        default: "Check for reduction operation"
      condition:
        default: "&{variable}-6"
      forward:
        default: "yes"
    out: [experiment]

  Import_variable:
    run: tasks/importncs.cwl
    in:
      experiment: Check_for_reduction_operation/experiment
      name:
        default: "Import variable"
      imp_dim:
        default: "time"
      measure:
        default: "@variable"
      src_path:
        default: "/home/jovyan/data/CMIP6/ScenarioMIP/CMCC/@{model}/@{scenario}/r1i1p1f1/@{frequency_&{variable}}/@{variable}/gn/*/@{variable}_@{frequency_&{variable}}_@{model}_@{scenario}_r1i1p1f1_gn*.nc"
      container: container
      subset_dims:
        default: "time"
      subset_filter: time_range
      subset_type:
        default: "coord"
      nfrag: nthreads
      nthreads: nthreads
    out: [experiment]
  Reduction_on_octets:
    run: tasks/reduce2.cwl
    in:
      experiment: Import_variable/experiment
      name:
        default: "Reduction on octets"
      operation:
        default: "median"
      concept_level:
        default: "o"
    out: [experiment]

  Else:
    run: tasks/else.cwl
    in:
      experiment: Check_for_reduction_operation/experiment
      name:
        default: "Else"
    out: [experiment]

  Import_sftlf:
    run: tasks/importncs.cwl
    in:
      experiment: Else/experiment
      name:
        default: "Import sftlf"
      measure:
        default: "@variable"
      src_path:
        default: "/home/jovyan/data/CMIP6/ScenarioMIP/CMCC/@{model}/@{scenario}/r1i1p1f1/@{frequency_&{variable}}/@{variable}/gn/*/@{variable}_@{frequency_&{variable}}_@{model}_@{scenario}_r1i1p1f1_gn.nc"
      container: container
      nfrag:
        default: 1
    out: [experiment]

  End_check:
    run: tasks/endif.cwl
    in:
      experiment: [Reduction_on_octets/experiment, Import_sftlf/experiment]
      name:
        default: "End check"
    out: [experiment]

  Rename_measure:
    run: tasks/apply.cwl
    in:
      experiment: End_check/experiment
      name:
        default: "Rename measure"
      measure:
        default: "@{measure_&{variable}}"
    out: [experiment]
  Export_variable:
    run: tasks/exportnc2.cwl
    in:
      experiment: Rename_measure/experiment
      name:
        default: "Export variable"
      output:
        default: "/home/jovyan/work/fires/output/@{variable}_@{frequency_&{variable}}_@{model}_@{scenario}_r1i1p1f1_gn.nc"
    out: [experiment]
  Regrid_variable:
    run: tasks/generic.cwl
    in:
      experiment:
        source: Export_variable/experiment
        valueFrom: ${ return [ self ]; }
      name:
        default: "Regrid variable"
      command:
        default: "/home/jovyan/work/fires/regrid.sh"
      args:
        default: "-90:90 0:360 r360x180 @{measure_&{variable}}"
      input:
        default: "/home/jovyan/work/fires/output/@{variable}_@{frequency_&{variable}}_@{model}_@{scenario}_r1i1p1f1_gn.nc"
      output:
        default: "/home/jovyan/work/fires/output/regridded_@{model}_@{scenario}.nc"
    out: [experiment]

  End_iteration_on_variables:
    run: tasks/endfor.cwl
    in:
      experiment:
        source: Regrid_variable/experiment
        valueFrom: ${ return [ self ]; }
      name:
        default: "End iteration on variables"
    out: [experiment]

  Infer_data:
    run: tasks/generic.cwl
    in:
      experiment:
        source: End_iteration_on_variables/experiment
        valueFrom: ${ return [ self ]; }
      name:
        default: "Infer data"
      command:
        default: "/home/jovyan/work/fires/fires.sh"
      input:
        default: "/home/jovyan/work/fires/output/regridded_@{model}_@{scenario}.nc"
      output:
        default: "/home/jovyan/work/fires/output/fires_@{model}_@{scenario}.nc"
      on_error:
        default: "repeat 10"
    out: [experiment]
  Import_model:
    run: tasks/importnc2.cwl
    in:
      experiment: Infer_data/experiment
      name:
        default: "Import model"
      imp_dim:
        default: "time"
      measure:
        default: "tos"
      src_path:
        default: "/home/jovyan/work/fires/output/fires_@{model}_@{scenario}.nc"
      container: container
      imp_concept_level:
        default: "o"
      nfrag: nthreads
      nthreads: nthreads
    out: [experiment]
  Reduction_on_years:
    run: tasks/reduce2.cwl
    in:
      experiment: Import_model/experiment
      name:
        default: "Reduction on years"
      operation:
        default: "avg"
      concept_level:
        default: "y"
    out: [experiment]
  Apply_the_mask:
    run: tasks/intercube.cwl
    in:
      experiment1: Reduction_on_years/experiment
      experiment2: Import_mask/experiment
      name:
        default: "Apply the mask"
      operation:
        default: "mul"
      extension_type:
        default: "append"
    out: [experiment]
  Export_model:
    run: tasks/exportnc2.cwl
    in:
      experiment: Apply_the_mask/experiment
      name:
        default: "Export model"
      output:
        default: "/home/jovyan/work/fires/output/inferenced_@{model}_@{scenario}.nc"
    out: [experiment]

  End_iteration_on_models:
    run: tasks/endfor.cwl
    in:
      experiment: [Apply_the_mask/experiment, Export_model/experiment]
      name:
        default: "End iteration on models"
      dependencies:
        default: "cube,"
    out: [experiment]

  Merge_models:
    run: tasks/mergecubes2.cwl
    in:
      experiment: End_iteration_on_models/experiment
      name:
        default: "Merge models"
      dim:
        default: "ensemble"
    out: [experiment]

  Iterate_on_ensemble_operations:
    run: tasks/for.cwl
    in:
      experiment: Merge_models/experiment
      name:
        default: "Iterate on ensemble operations"
      parallel:
        default: "yes"
      key:
        default: "operation"
      values:
        default: "avg|min|max|var|std"
    out: [experiment]
  Ensemble_operation:
    run: tasks/reduce2.cwl
    in:
      experiment: Iterate_on_ensemble_operations/experiment
      name:
        default: "Ensemble operation"
      operation:
        default: "@{operation}"
      dim:
        default: "ensemble"
    out: [experiment]
  Export_scenario:
    run: tasks/exportnc2.cwl
    in:
      experiment: Ensemble_operation/experiment
      name:
        default: "Export scenario"
      output:
        default: "/home/jovyan/work/fires/output/@{operation}_@{scenario}.nc"
    out: [experiment]
  End_iteration_on_ensemble_operations:
    run: tasks/endfor.cwl
    in:
      experiment:
        source: Export_scenario/experiment
        valueFrom: ${ return [ self ]; }
      name:
        default: "End iteration on ensemble operations"
    out: [experiment]

  End_iteration_on_scenarios:
    run: tasks/endfor.cwl
    in:
      experiment:
        source: End_iteration_on_ensemble_operations/experiment
        valueFrom: ${ return [ self ]; }
      name:
        default: "End iteration on scenarios"
    out: [experiment]

