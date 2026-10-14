import numpy as np
import pytest

from src.data_loader import load_metabric

PREAMBLE = "#Patient Identifier\n#Description\n#Datatype\n#Priority\n"


def write_study(folder):
    """A four-patient study in the cBioPortal file layout."""
    folder.mkdir()
    (folder / "data_clinical_patient.txt").write_text(
        PREAMBLE
        + "PATIENT_ID\tAGE_AT_DIAGNOSIS\tNPI\tOS_MONTHS\tOS_STATUS\tRFS_MONTHS\tRFS_STATUS\n"
        + "P1\t50\t3.0\t100.0\t1:DECEASED\t40.0\t1:Recurred\n"
        + "P2\t60\t4.0\t120.0\t0:LIVING\t120.0\t0:Not Recurred\n"
        + "P3\t70\t5.0\t80.0\t1:DECEASED\t80.0\t0:Not Recurred\n"
        + "P4\t45\t2.0\t90.0\t0:LIVING\t\t\n"
    )
    (folder / "data_clinical_sample.txt").write_text(
        PREAMBLE
        + "PATIENT_ID\tSAMPLE_ID\tTUMOR_SIZE\tGRADE\tER_IHC\n"
        + "".join(f"P{i}\tS{i}\t20\t2\tPositve\n" for i in range(1, 5))
    )
    (folder / "data_mrna_illumina_microarray.txt").write_text(
        "Hugo_Symbol\tEntrez_Gene_Id\tS1\tS2\tS3\tS4\n"
        "GENE_A\t1\t5.0\t6.0\t7.0\t8.0\n"
        "GENE_B\t2\t1.0\t3.0\t2.0\t4.0\n"
    )
    return folder


def test_rfs_endpoint_reads_relapse_columns(tmp_path):
    study = write_study(tmp_path / "study")
    data = load_metabric(study, endpoint="rfs", use_cache=False)

    # P4 has no relapse record, so it is dropped for RFS but kept for OS.
    assert list(data.sample_ids) == ["S1", "S2", "S3"]
    np.testing.assert_allclose(data.time, [40.0, 120.0, 80.0])
    np.testing.assert_allclose(data.event, [1.0, 0.0, 0.0])

    overall = load_metabric(study, endpoint="os", use_cache=False)
    assert list(overall.sample_ids) == ["S1", "S2", "S3", "S4"]
    np.testing.assert_allclose(overall.event, [1.0, 0.0, 1.0, 0.0])


def test_unknown_endpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown endpoint"):
        load_metabric(tmp_path, endpoint="dfs", use_cache=False)
