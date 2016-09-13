<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();
$db = db_connect();

$my_userid = $_SESSION[check_valid_user()];
$my_patient_info = patient::get_patient_info($db, $my_userid);
$my_patientid = $my_patient_info['patientid'];

display_my_wl_status($db, $my_patientid);

$db->close();

do_footer();

echo '<meta http-equiv="refresh" content="60">';
?>