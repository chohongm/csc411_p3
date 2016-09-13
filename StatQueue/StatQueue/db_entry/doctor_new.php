<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
session_start();

if (!filled_out($_POST)){
  	throw new Exception("You have not entered all the required details.<br />
  						 Please go back and try again.");
}

if (!valid_birthdate($_POST['birthdate'])){
	throw new Exception("Invalid birthdate.");
}

if (!valid_email($_POST['email'])){
	throw new Exception("The email you have entered is not valid.");
}

if (!get_magic_quotes_gpc()){
	foreach ($_POST as $key => $value){
		addslashes($value);
	}
}

$doctorname = $_POST['firstname']." ".$_POST['lastname'];
$doctor_info['doctorname'] = $doctorname;
array_splice($_POST, 0, 2);
$doctor_info = array_merge($doctor_info, $_POST);

@ $db = db_connect();

if (doctor::lookup_doctor($db, $doctorname)){
	throw new Exception('The doctor you are trying to register already exists in doctors database.');
} else {
	$userid = $_SESSION[check_valid_user()];
	$clinicid = admin::get_clinicid($db, $userid);
	$doctor_info['clinicid'] = $clinicid;
	
	$new_doctor = new doctor($doctor_info);
	
	$new_doctor->register_doctor();

	include("$DOCUMENT_ROOT/db_entry/doctors_profile.php");
}
?>