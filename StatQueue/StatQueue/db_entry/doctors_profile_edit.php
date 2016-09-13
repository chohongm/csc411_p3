<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once ("$DOCUMENT_ROOT/functions/statq_fns.php");

if (!filled_out($_POST)){
	throw new Exception('Please go back and enter every required details.');
}

if (isset($_POST['email'])){
	if (!valid_email($_POST['email'])){
		throw new Exception("The email you have entered is not valid.");
	}
}

$db = db_connect();

$doctorid = array_pop($_POST);

$edit_list = array();
foreach ($_POST as $key => $value){
	$value = addslashes($value);
	$edit_list[$key] = $value;
}

if (count($edit_list) != 0){
	doctor::update_doctor($db, $doctorid, $edit_list);
}
$_GET['profile'] = $doctorid;
include("$DOCUMENT_ROOT/db_entry/doctors_profile_form.php");
?>