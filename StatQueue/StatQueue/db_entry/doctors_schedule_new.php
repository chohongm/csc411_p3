<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once ("$DOCUMENT_ROOT/functions/statq_fns.php");

if (!filled_out($_POST)){
	throw new Exception('Please go back and enter every required details.');
}

$db = db_connect();
$doctorid = array_pop($_POST);

$values = array();
foreach ($_POST as $value){
	array_push($values, $value);
}

$schedule = array();
for ($i = 0; $i < count($values); $i+=2){
	if($values[$i] > $values[$i+1]){
		throw new Exception('Entered hours are incorrectly matched. Make sure hours in second
							 box are greater than hours in first box.');
	}
	$string ='from '.$values[$i].' to '.$values[$i+1];
	array_push($schedule, $string);
}

doctor::set_schedule($db, $doctorid, $schedule);
doctor::set_available($db, $doctorid);

$_GET['profile'] = $doctorid;
include("$DOCUMENT_ROOT/db_entry/doctors_profile_form.php");

?>