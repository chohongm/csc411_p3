<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

if(isset($_GET['patientid'])){
	$patientid = $_GET['patientid'];
	$patient_profile = new patient_profile($patientid);
	$_SESSION['patient_profile'] = $patient_profile;
}

$patient_profile->display_profile();

do_footer();
?>