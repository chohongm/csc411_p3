<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

$db = db_connect();

$clinicid = $_GET['clinicid'];
$wl_id = $_GET['wl_id'];
$wl_length = $_GET['wl_length'];
clinic::unlist_patient($db, $clinicid, $wl_id, $wl_length);
echo "<h3 style = 'text-align: center'>You are successfully removed from waitlist.</h3>";
echo "<p style = 'text-align: center'><span class='spec'>Now loading 'my waitlist status' page..</span></p>";
echo '<meta http-equiv="refresh" content= "2;URL= \'/clinic/my_wl_status.php\'">';

$db->close();

do_footer();