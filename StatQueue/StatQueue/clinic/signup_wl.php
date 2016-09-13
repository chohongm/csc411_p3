<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");

session_start();

$db = db_connect();

$userid = $_SESSION[check_valid_user()];
$patient_info = patient::get_patient_info($db, $userid);
$patientid = $patient_info['patientid'];
$query = "select * from waitlist where patientid = ".$patientid;
$result = mysqli_query($db, $query);

if($result->num_rows >= 2){
	throw new Exception("Users are not allowed to sign-up for more 
						than two waitlists at the same time.");
}
$clinicid = $_POST['clinicid'];
$clinic = new clinic($db, $clinicid);

if ($clinic->lookup_patient($patientid)){
	throw new Exception("You are already listed on this clinic's waitlist.");
}

if (!get_magic_quotes_gpc()){
	foreach ($_POST as $key => $value){
		addslashes($value);
	}
}

if (isset($_POST['doctorid'])){
	$doctorid = $_POST['doctorid'];
	$clinicid = $_POST['clinicid'];
	$doctor_info = doctor::get_doctor_info($db, $doctorid);
	$_POST['doctor'] = $doctor_info['doctorname'];
} else if (isset($_POST['clinicid'])){
	$clinicid = $_POST['clinicid'];
	$_POST['doctor'] = find_term_like($db, 'doctorname', 'doctors', $_POST['doctor']);
} 

foreach ($patient_info as $key => $value){
	if ($key == 'patientname' || $key == 'birthdate' ||
		$key == 'gender' || $key == 'medicalhistory'){
		$wl_info[$key] = $value;
	}
}
$wl_info['treatment'] = $_POST['treatment'];
$wl_info['doctor'] = $_POST['doctor'];
$wl_info['patientid'] = $patientid;

$clinic->list_patient($wl_info);

include("$DOCUMENT_ROOT/clinic/my_wl_status.php");

?>