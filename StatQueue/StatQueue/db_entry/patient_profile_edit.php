<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
session_start();

if (!filled_out($_POST)){
	throw new Exception("You have not entered all the required details.<br />"
			."Please go back and try again.");
}

if (isset($_POST['email'])){
	if (!valid_email($_POST['email'])){
		throw new Exception("The email you have entered is not valid.");
	}
}

if (isset($_POST['postalcode'])){
	if (!valid_postalcode($_POST['postalcode'])){
		throw new Exception("Postal code you have entered is not valid.");
	}
}

$db = db_connect();

$patient_profile = $_SESSION['patient_profile'];
$patient_profile->db = $db;
$patientid = $patient_profile->patientid;

$edit_list = array();

foreach ($_POST as $key => $value){
	$value = addslashes($value);
	$edit_list[$key] = $value;
}

$result = patient::update_patient($db, $patientid, $edit_list);
if ($result){
	$patient_profile = new patient_profile($patientid);
	$_SESSION['patient_profile'] = $patient_profile;
	include('patient_profile.php');
}
?>