# Test-set failures

## wrong_label (164)
- Criterion (inclusion): Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)
- Expert label: not included. Predicted: included.
- Expert sentences: []. Predicted sentences: [0].
- Quoted expert evidence: []
- Quoted predicted evidence: ['0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.']
- The predicted label does not match the expert label.

## missing_information (169)
- Criterion (exclusion): Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- Expert label: not enough information. Predicted: not excluded.
- Expert sentences: []. Predicted sentences: [].
- Quoted expert evidence: []
- Quoted predicted evidence: []
- The expert label is not enough information. A missing fact is not treated as eligibility or ineligibility.

## missed_gold_sentence (185)
- Criterion (inclusion): Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.
- Expert label: included. Predicted: not enough information.
- Expert sentences: [2, 3]. Predicted sentences: [].
- Quoted expert evidence: ['2. On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.', '3. Her body temperature is normal and her blood pressure is 80/60.']
- Quoted predicted evidence: []
- The predicted label does not match the expert label.

## empty_gold_but_model_cited (161)
- Criterion (inclusion): Patients undergoing elective carotid endarterectomy
- Expert label: not included. Predicted: not included.
- Expert sentences: []. Predicted sentences: [0].
- Quoted expert evidence: []
- Quoted predicted evidence: ['0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.']
- The expert cited nothing, and the model cited a sentence. That citation is not treated as proof.

## not_applicable_confusion (683)
- Criterion (exclusion): Previous tonsillectomy, including partial tonsillectomy
- Expert label: not applicable. Predicted: not excluded.
- Expert sentences: []. Predicted sentences: [].
- Quoted expert evidence: []
- Quoted predicted evidence: []
- The predicted label does not match the expert label.

## additional_mismatch (162)
- Criterion (inclusion): Patients undergoing open abdominal aortic aneurysm repair
- Expert label: not included. Predicted: not included.
- Expert sentences: []. Predicted sentences: [0].
- Quoted expert evidence: []
- Quoted predicted evidence: ['0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.']
- The cited sentence ids do not match the expert citation.

## additional_mismatch (163)
- Criterion (inclusion): Patients undergoing endovascular abdominal aneurysm repair
- Expert label: not included. Predicted: not included.
- Expert sentences: []. Predicted sentences: [0].
- Quoted expert evidence: []
- Quoted predicted evidence: ['0. A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.']
- The cited sentence ids do not match the expert citation.

## additional_mismatch (166)
- Criterion (exclusion): Significant upper limb peripheral arterial disease
- Expert label: not excluded. Predicted: not excluded.
- Expert sentences: []. Predicted sentences: [1].
- Quoted expert evidence: []
- Quoted predicted evidence: ['1. She is now complaining of a cool right foot.']
- The cited sentence ids do not match the expert citation.

## truncation_would_drop_gold (160)
- Criterion (inclusion): Patient willing to give full informed consent for participation
- Expert label: included. Predicted: included.
- Expert sentences: [3]. Predicted sentences: [3].
- Quoted expert evidence: ['3. The patient will provide informed consent, and will comply with the trial protocol without any practical issues.']
- Quoted predicted evidence: ['3. The patient will provide informed consent, and will comply with the trial protocol without any practical issues.']
- At the trained maximum length of 1024, no expert sentence id was dropped. If the prompt budget were 256 tokens, expert sentence ids [3] would fall outside the kept prefix.
